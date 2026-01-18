import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
@property
def __path__(self):
    return []