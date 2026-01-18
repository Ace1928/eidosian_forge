import operator
import sys
from functools import reduce
from importlib import import_module
from types import ModuleType
def __evaluated__(self):
    try:
        object.__getattribute__(self, '__thing')
    except AttributeError:
        return False
    return True