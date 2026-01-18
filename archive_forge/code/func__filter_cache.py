import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
def _filter_cache(self, dmap, kdims):
    """
        Returns a filtered version of the DynamicMap cache leaving only
        keys consistently with the newly specified values
        """
    filtered = []
    for key, value in dmap.data.items():
        if not any((kd.values and v not in kd.values for kd, v in zip(kdims, key))):
            filtered.append((key, value))
    return filtered