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
def _raise_on_invalid_property_error(self, _error_to_raise=None):
    """
        Returns a function that raises informative exception when invalid
        property names are encountered. The _error_to_raise argument allows
        specifying the exception to raise, which is ValueError if None.

        Parameters
        ----------
        args : list[str]
            List of property names that have already been determined to be
            invalid

        Raises
        ------
        ValueError by default, or _error_to_raise if not None
        """
    if _error_to_raise is None:
        _error_to_raise = ValueError

    def _ret(*args):
        invalid_props = args
        if invalid_props:
            if len(invalid_props) == 1:
                prop_str = 'property'
                invalid_str = repr(invalid_props[0])
            else:
                prop_str = 'properties'
                invalid_str = repr(invalid_props)
            module_root = 'plotly.graph_objs.'
            if self._parent_path_str:
                full_obj_name = module_root + self._parent_path_str + '.' + self.__class__.__name__
            else:
                full_obj_name = module_root + self.__class__.__name__
            guessed_prop = None
            if len(invalid_props) == 1:
                try:
                    guessed_prop = find_closest_string(invalid_props[0], self._valid_props)
                except Exception:
                    pass
            guessed_prop_suggestion = ''
            if guessed_prop is not None:
                guessed_prop_suggestion = 'Did you mean "%s"?' % (guessed_prop,)
            raise _error_to_raise('Invalid {prop_str} specified for object of type {full_obj_name}: {invalid_str}\n\n{guessed_prop_suggestion}\n\n    Valid properties:\n{prop_descriptions}\n{guessed_prop_suggestion}\n'.format(prop_str=prop_str, full_obj_name=full_obj_name, invalid_str=invalid_str, prop_descriptions=self._prop_descriptions, guessed_prop_suggestion=guessed_prop_suggestion))
    return _ret