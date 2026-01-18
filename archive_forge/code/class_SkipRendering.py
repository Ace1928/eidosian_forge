import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
class SkipRendering(Exception):
    """
    A SkipRendering exception in the plotting code will make the display
    hooks fall back to a text repr. Used to skip rendering of
    DynamicMaps with exhausted element generators.
    """

    def __init__(self, message='', warn=True):
        self.warn = warn
        super().__init__(message)