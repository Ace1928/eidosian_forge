import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence
class Exchangeable_simulator(GEE_simulator):
    """
    Simulate exchangeable Poisson data.

    The data within a cluster are simulated as y_i = z_c + z_i.  The
    z_c, and {z_i} are independent Poisson random variables with
    expected values e_c and {e_i}, respectively.  In order for the
    pairwise correlation to be equal to `f` for all pairs, we need

         e_c / sqrt((e_c + e_i) * (e_c + e_j)) = f for all i, j.

    By setting all e_i = e within a cluster, these equations can be
    satisfied.  We thus need

         e_c * (1 - f) = f * e,

    which can be solved (non-uniquely) for e and e_c.
    """
    scale_inv = 1.0

    def print_dparams(self, dparams_est):
        OUT.write('Estimated common pairwise correlation:   %8.4f\n' % dparams_est[0])
        OUT.write('True common pairwise correlation:        %8.4f\n' % self.dparams[0])
        OUT.write('Estimated inverse scale parameter:       %8.4f\n' % dparams_est[1])
        OUT.write('True inverse scale parameter:            %8.4f\n' % self.scale_inv)
        OUT.write('\n')

    def simulate(self):
        endog, exog, group, time = ([], [], [], [])
        f = np.sum(self.params ** 2)
        u, s, vt = np.linalg.svd(np.eye(len(self.params)) - np.outer(self.params, self.params) / f)
        params0 = u[:, np.flatnonzero(s > 1e-06)]
        for i in range(self.ngroups):
            gsize = np.random.randint(self.group_size_range[0], self.group_size_range[1])
            group.append([i] * gsize)
            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)
            e_c = np.random.uniform(low=1, high=10)
            e = e_c * (1 - self.dparams[0]) / self.dparams[0]
            common = np.random.poisson(e_c)
            unique = np.random.poisson(e, gsize)
            endog1 = common + unique
            endog.append(endog1)
            lpr = np.log(e_c + e) * np.ones(gsize)
            exog1 = np.outer(lpr, self.params) / np.sum(self.params ** 2)
            emat = np.random.normal(size=(len(lpr), params0.shape[1]))
            exog1 += np.dot(emat, params0.T)
            exog.append(exog1)
        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)