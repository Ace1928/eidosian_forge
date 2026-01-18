from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator
class ordinal_simulator(GEE_simulator):
    thresholds = None

    def true_params(self):
        return np.concatenate((self.thresholds, self.params))

    def starting_values(self, nconstraints):
        beta = gee_ordinal_starting_values(self.endog, len(self.params))
        if nconstraints > 0:
            m = self.exog_ex.shape[1] - nconstraints
            beta = beta[0:m]
        return beta

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
            exog1 = np.random.normal(size=(gsize, len(self.params)))
            exog.append(exog1)
            lp = np.dot(exog1, self.params)
            z = np.random.uniform(size=gsize)
            z = np.log(z / (1 - z)) + lp
            endog1 = np.array([np.sum(x > self.thresholds) for x in z])
            endog.append(endog1)
        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)
        self.offset = np.zeros(len(self.endog), dtype=np.float64)