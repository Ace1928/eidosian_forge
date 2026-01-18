import numpy as np
from . import _odepack
from copy import copy
import warnings
class ODEintWarning(Warning):
    """Warning raised during the execution of `odeint`."""
    pass