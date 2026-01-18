import logging
import glob
from pyomo.common.tempfiles import TempfileManager
from pyomo.solvers.plugins.solvers.ASL import ASL
def _initialize_callbacks(self, model):
    self._model = model
    self._model._gjh_info = None
    super()._initialize_callbacks(model)