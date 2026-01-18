import re
import inspect
import os
import sys
from importlib.machinery import SourceFileLoader
def __dictify__(self, obj, prefix):
    """
        Private helper method for to_dict.
        """
    for k, v in obj.copy().items():
        if prefix:
            del obj[k]
            k = '%s%s' % (prefix, k)
        if isinstance(v, Config):
            v = self.__dictify__(dict(v), prefix)
        obj[k] = v
    return obj