from pyomo.common.dependencies import numpy as np, pandas as pd, matplotlib as plt
from pyomo.core.expr.numvalue import value
from itertools import product
import logging
from pyomo.opt import SolverStatus, TerminationCondition
def _print_FIM_info(self, FIM):
    """
        using a dictionary to store all FIM information

        Parameters
        ----------
        FIM: the Fisher Information Matrix, needs to be P.D. and symmetric

        Returns
        -------
        fim_info: a FIM dictionary containing the following key:value pairs
            ~['FIM']: a list of FIM itself
            ~[design variable name]: a list of design variable values at each time point
            ~['Trace']: a scalar number of Trace
            ~['Determinant']: a scalar number of determinant
            ~['Condition number:']: a scalar number of condition number
            ~['Minimal eigen value:']: a scalar number of minimal eigen value
            ~['Eigen values:']: a list of all eigen values
            ~['Eigen vectors:']: a list of all eigen vectors
        """
    eig = np.linalg.eigvals(FIM)
    self.FIM = FIM
    self.trace = np.trace(FIM)
    self.det = np.linalg.det(FIM)
    self.min_eig = min(eig)
    self.cond = max(eig) / min(eig)
    self.eig_vals = eig
    self.eig_vecs = np.linalg.eig(FIM)[1]
    self.logger.info('FIM: %s; \n Trace: %s; \n Determinant: %s;', self.FIM, self.trace, self.det)
    self.logger.info('Condition number: %s; \n Min eigenvalue: %s.', self.cond, self.min_eig)