import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
def _mirror_name(self, fullname):
    """get the name of the mirrored module"""
    return self.mirror + fullname[len(self.src):]