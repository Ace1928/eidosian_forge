import ast
from functools import lru_cache, reduce
from numbers import Real
import operator
import os
import re
import numpy as np
from matplotlib import _api, cbook
from matplotlib.cbook import ls_mapper
from matplotlib.colors import Colormap, is_color_like
from matplotlib._fontconfig_pattern import parse_fontconfig_pattern
from matplotlib._enums import JoinStyle, CapStyle
from cycler import Cycler, cycler as ccycler
def _validate_toolbar(s):
    s = ValidateInStrings('toolbar', ['None', 'toolbar2', 'toolmanager'], ignorecase=True)(s)
    if s == 'toolmanager':
        _api.warn_external('Treat the new Tool classes introduced in v1.5 as experimental for now; the API and rcParam may change in future versions.')
    return s