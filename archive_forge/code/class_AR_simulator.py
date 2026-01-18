from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
class AR_simulator(GEE_simulator):
    distfun = [lambda x, y: np.sqrt(np.sum((x - y) ** 2))]

    def print_dparams(self, dparams_est):
        OUT.write('AR coefficient estimate:   %8.4f\n' % dparams_est[0])
        OUT.write('AR coefficient truth:      %8.4f\n' % self.dparams[0])
        OUT.write('Error variance estimate:   %8.4f\n' % dparams_est[1])
        OUT.write('Error variance truth:      %8.4f\n' % self.error_sd ** 2)
        OUT.write('\n')

    def simulate(self):
        endog, exog, group, time = ([], [], [], [])
        for i in range(self.ngroups):
            gsize = np.random.randint(self.group_size_range[0], self.group_size_range[1])
            group.append([i] * gsize)
            time1 = np.random.normal(size=(gsize, 2))
            time.append(time1)
            exog1 = np.random.normal(size=(gsize, 5))
            exog1[:, 0] = 1
            exog.append(exog1)
            distances = scipy.spatial.distance.cdist(time1, time1, self.distfun[0])
            correlations = self.dparams[0] ** distances
            correlations_sr = np.linalg.cholesky(correlations)
            errors = np.dot(correlations_sr, np.random.normal(size=gsize))
            endog1 = np.dot(exog1, self.params) + errors * self.error_sd
            endog.append(endog1)
        self.exog = np.concatenate(exog, axis=0)
        self.endog = np.concatenate(endog)
        self.time = np.concatenate(time, axis=0)
        self.group = np.concatenate(group)