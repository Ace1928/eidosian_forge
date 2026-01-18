from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator
class nominal_simulator(GEE_simulator):

    def starting_values(self, nconstraints):
        return None

    def true_params(self):
        return np.concatenate(self.params[:-1])

    def print_dparams(self, dparams_est):
        OUT.write('Odds ratio estimate:   %8.4f\n' % dparams_est[0])
        OUT.write('Odds ratio truth:      %8.4f\n' % self.dparams[0])
        OUT.write('\n')

    def simulate(self):
        endog, exog, group, time = ([], [], [], [])
        for i in range(self.ngroups):
            gsize = np.random.randint(self.group_size_range[0], self.group_size_range[1])
            group.append([i] * gsize)
            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)
            exog1 = np.random.normal(size=(gsize, len(self.params[0])))
            exog.append(exog1)
            prob = [np.exp(np.dot(exog1, p)) for p in self.params]
            prob = np.vstack(prob).T
            prob /= prob.sum(1)[:, None]
            m = len(self.params)
            endog1 = []
            for k in range(gsize):
                pdist = stats.rv_discrete(values=(lrange(m), prob[k, :]))
                endog1.append(pdist.rvs())
            endog.append(np.asarray(endog1))
        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog).astype(np.int32)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)
        self.offset = np.zeros(len(self.endog), dtype=np.float64)