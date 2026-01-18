import abc
import logging
from pyomo.environ import SolverFactory
class CplexConflict(_IISBase):

    def compute(self):
        self._solver._solver_model.conflict.refine()

    def write(self, file_name):
        self._solver._solver_model.conflict.write(file_name)
        return file_name