import warnings
from collections import namedtuple
import operator
from . import _zeros
from ._optimize import OptimizeResult, _call_callback_maybe_halt
import numpy as np
Terminate due to convergence, non-finite values, or error increase