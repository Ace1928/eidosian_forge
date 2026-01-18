import abc
import logging
from pyomo.environ import SolverFactory
class XpressIIS(_IISBase):

    def compute(self):
        self._solver._solver_model.iisfirst(1)

    def write(self, file_name):
        self._solver._solver_model.iiswrite(0, file_name, 0, 'l')
        if self._solver._version[0] < 38:
            return file_name
        else:
            return _remove_suffix(file_name, '.lp') + '.lp'