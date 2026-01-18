import collections
import copy
import itertools
import random
import re
import warnings
class TreeElement:
    """Base class for all Bio.Phylo classes."""

    def __repr__(self) -> str:
        """Show this object's constructor with its primitive arguments."""

        def pair_as_kwarg_string(key, val):
            if isinstance(val, str):
                val = val[:57] + '...' if len(val) > 60 else val
                return f"{key}='{val}'"
            return f'{key}={val}'
        return '%s(%s)' % (self.__class__.__name__, ', '.join((pair_as_kwarg_string(key, val) for key, val in sorted(self.__dict__.items()) if val is not None and type(val) in (str, int, float, bool, str))))

    def __str__(self) -> str:
        return self.__repr__()