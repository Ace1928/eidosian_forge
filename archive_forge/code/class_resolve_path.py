import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class resolve_path(ParameterizedFunction):
    """
    Find the path to an existing file, searching the paths specified
    in the search_paths parameter if the filename is not absolute, and
    converting a UNIX-style path to the current OS's format if
    necessary.

    To turn a supplied relative path into an absolute one, the path is
    appended to paths in the search_paths parameter, in order, until
    the file is found.

    An IOError is raised if the file is not found.

    Similar to Python's os.path.abspath(), except more search paths
    than just os.getcwd() can be used, and the file must exist.
    """
    search_paths = List(default=[os.getcwd()], pickle_default_value=False, doc='\n        Prepended to a non-relative path, in order, until a file is\n        found.')
    path_to_file = Boolean(default=True, pickle_default_value=False, allow_None=True, doc="\n        String specifying whether the path refers to a 'File' or a\n        'Folder'. If None, the path may point to *either* a 'File' *or*\n        a 'Folder'.")

    def __call__(self, path, **params):
        p = ParamOverrides(self, params)
        path = os.path.normpath(path)
        ftype = 'File' if p.path_to_file is True else 'Folder' if p.path_to_file is False else 'Path'
        if not p.search_paths:
            p.search_paths = [os.getcwd()]
        if os.path.isabs(path):
            if p.path_to_file is None and os.path.exists(path) or (p.path_to_file is True and os.path.isfile(path)) or (p.path_to_file is False and os.path.isdir(path)):
                return path
            raise OSError(f"{ftype} '{path}' not found.")
        else:
            paths_tried = []
            for prefix in p.search_paths:
                try_path = os.path.join(os.path.normpath(prefix), path)
                if p.path_to_file is None and os.path.exists(try_path) or (p.path_to_file is True and os.path.isfile(try_path)) or (p.path_to_file is False and os.path.isdir(try_path)):
                    return try_path
                paths_tried.append(try_path)
            raise OSError(ftype + ' ' + os.path.split(path)[1] + ' was not found in the following place(s): ' + str(paths_tried) + '.')