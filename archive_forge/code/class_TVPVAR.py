import os
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from statsmodels import datasets
from statsmodels.tools import add_constant
from statsmodels.tsa.tsatools import lagmat
from statsmodels.tsa.statespace import mlemodel
class TVPVAR(mlemodel.MLEModel):

    def __init__(self, endog):
        if not isinstance(endog, pd.DataFrame):
            endog = pd.DataFrame(endog)
        k = endog.shape[1]
        augmented = lagmat(endog, 1, trim='both', original='in', use_pandas=True)
        endog = augmented.iloc[:, :k]
        exog = add_constant(augmented.iloc[:, k:])
        k_states = k * (k + 1)
        super().__init__(endog, k_states=k_states)
        self.ssm.initialize('known', stationary_cov=np.eye(self.k_states) * 5)
        self['design'] = np.zeros((self.k_endog, self.k_states, self.nobs))
        for i in range(self.k_endog):
            start = i * (self.k_endog + 1)
            end = start + self.k_endog + 1
            self['design', i, start:end, :] = exog.T
        self['transition'] = np.eye(k_states)
        self['selection'] = np.eye(k_states)
        self._obs_cov_slice = np.s_[:self.k_endog * (self.k_endog + 1) // 2]
        self._obs_cov_tril = np.tril_indices(self.k_endog)
        self._state_cov_slice = np.s_[-self.k_states:]
        self._state_cov_ix = ('state_cov',) + np.diag_indices(self.k_states)

    @property
    def state_names(self):
        state_names = []
        for i in range(self.k_endog):
            endog_name = self.endog_names[i]
            state_names += ['intercept.%s' % endog_name]
            state_names += ['L1.{}->{}'.format(other_name, endog_name) for other_name in self.endog_names]
        return state_names

    def update_direct(self, obs_cov, state_cov_diag):
        self['obs_cov'] = obs_cov
        self[self._state_cov_ix] = state_cov_diag