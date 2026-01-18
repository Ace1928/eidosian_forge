import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _restore_dispatch(name):
    return _registered_algorithms[name]