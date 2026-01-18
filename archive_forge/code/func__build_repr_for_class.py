import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
@staticmethod
def _build_repr_for_class(props, class_name, parent_path_str=None):
    """
        Helper to build representation string for a class

        Parameters
        ----------
        class_name : str
            Name of the class being represented
        parent_path_str : str of None (default)
            Name of the class's parent package to display
        props : dict
            Properties to unpack into the constructor

        Returns
        -------
        str
            The representation string
        """
    from plotly.utils import ElidedPrettyPrinter
    if parent_path_str:
        class_name = parent_path_str + '.' + class_name
    if len(props) == 0:
        repr_str = class_name + '()'
    else:
        pprinter = ElidedPrettyPrinter(threshold=200, width=120)
        pprint_res = pprinter.pformat(props)
        body = '   ' + pprint_res[1:-1].replace('\n', '\n   ')
        repr_str = class_name + '({\n ' + body + '\n})'
    return repr_str