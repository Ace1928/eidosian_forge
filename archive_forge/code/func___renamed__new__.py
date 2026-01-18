import logging
import functools
import inspect
import itertools
import sys
import textwrap
import types
from pyomo.common.errors import DeveloperError
def __renamed__new__(cls, *args, **kwargs):
    cls.__renamed__warning__("Instantiating class '%s'." % (cls.__name__,))
    return new_class(*args, **kwargs)