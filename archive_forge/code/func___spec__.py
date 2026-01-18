import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
@property
def __spec__(self):
    """Don't produce __spec__ until requested"""
    return import_module(self._mirror).__spec__