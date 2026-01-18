from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _validate_state(self, state):
    supported_state = [key for key, value in self._state_modifier_keys.items() if key != 'clear' and value != 'not-applicable']
    _api.check_in_list(supported_state, state=state)