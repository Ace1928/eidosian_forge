import importlib
import os
import sys
import warnings
def _CanImport(mod_name):
    try:
        mod = importlib.import_module(mod_name)
        if not mod:
            raise ImportError(mod_name + ' import succeeded but was None')
        return True
    except ImportError:
        return False