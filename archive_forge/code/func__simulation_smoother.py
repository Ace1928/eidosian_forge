import numpy as np
from . import tools
@property
def _simulation_smoother(self):
    prefix = self.model.prefix
    if prefix in self._simulation_smoothers:
        return self._simulation_smoothers[prefix]
    return None