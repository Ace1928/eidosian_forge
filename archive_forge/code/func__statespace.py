import warnings
import numpy as np
from .tools import (
from .initialization import Initialization
from . import tools
@property
def _statespace(self):
    prefix = self.prefix
    if prefix in self._statespaces:
        return self._statespaces[prefix]
    return None