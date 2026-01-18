import collections
import importlib
import os
import pkgutil
def _tupleize(d):
    """Convert a dict of options to the 2-tuple format."""
    return [(key, value) for key, value in d.items()]