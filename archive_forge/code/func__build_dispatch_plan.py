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
def _build_dispatch_plan(key_path_strs):
    """
        Build a dispatch plan for a list of key path strings

        A dispatch plan is a dict:
           - *from* path tuples that reference an object that has descendants
             that are referenced in `key_path_strs`.
           - *to* sets of tuples that correspond to descendants of the object
             above.

        Parameters
        ----------
        key_path_strs : list[str]
            List of key path strings. For example:

            ['xaxis.rangeselector.font.color', 'xaxis.rangeselector.bgcolor']

        Returns
        -------
        dispatch_plan: dict[tuple[str|int], set[tuple[str|int]]]

        Examples
        --------

        >>> key_path_strs = ['xaxis.rangeselector.font.color',
        ...                  'xaxis.rangeselector.bgcolor']

        >>> BaseFigure._build_dispatch_plan(key_path_strs) # doctest: +SKIP
            {(): {'xaxis',
                  ('xaxis', 'rangeselector'),
                  ('xaxis', 'rangeselector', 'bgcolor'),
                  ('xaxis', 'rangeselector', 'font'),
                  ('xaxis', 'rangeselector', 'font', 'color')},
             ('xaxis',): {('rangeselector',),
                          ('rangeselector', 'bgcolor'),
                          ('rangeselector', 'font'),
                          ('rangeselector', 'font', 'color')},
             ('xaxis', 'rangeselector'): {('bgcolor',),
                                          ('font',),
                                          ('font', 'color')},
             ('xaxis', 'rangeselector', 'font'): {('color',)}}
        """
    dispatch_plan = {}
    for key_path_str in key_path_strs:
        key_path = BaseFigure._str_to_dict_path(key_path_str)
        key_path_so_far = ()
        keys_left = key_path
        for next_key in key_path:
            if key_path_so_far not in dispatch_plan:
                dispatch_plan[key_path_so_far] = set()
            to_add = [keys_left[:i + 1] for i in range(len(keys_left))]
            dispatch_plan[key_path_so_far].update(to_add)
            key_path_so_far = key_path_so_far + (next_key,)
            keys_left = keys_left[1:]
    return dispatch_plan