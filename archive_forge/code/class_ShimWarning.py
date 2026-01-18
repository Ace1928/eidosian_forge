import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
class ShimWarning(Warning):
    """A warning to show when a module has moved, and a shim is in its place."""