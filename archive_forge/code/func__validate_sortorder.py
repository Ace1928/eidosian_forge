import os
import sys
import _yappi
import pickle
import threading
import warnings
import types
import inspect
import itertools
from contextlib import contextmanager
def _validate_sortorder(sort_order):
    sort_order = sort_order.lower()
    if sort_order not in SORT_ORDERS:
        raise YappiError(f"Invalid SortOrder parameter: '{sort_order}'")
    return sort_order