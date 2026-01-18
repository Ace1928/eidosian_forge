import abc
import logging
from pyomo.environ import SolverFactory
class GurobiIIS(_IISBase):

    def compute(self):
        self._solver._solver_model.computeIIS()

    def write(self, file_name):
        file_name = _remove_suffix(file_name, '.ilp')
        file_name += '.ilp'
        self._solver._solver_model.write(file_name)
        return file_name