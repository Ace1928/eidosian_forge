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
class abbreviated_exception:
    """
    Context manager used to to abbreviate tracebacks using an
    AbbreviatedException when a backend may raise an error due to
    incorrect style options.
    """

    def __enter__(self):
        return self

    def __exit__(self, etype, value, traceback):
        if isinstance(value, Exception):
            raise AbbreviatedException(etype, value, traceback)