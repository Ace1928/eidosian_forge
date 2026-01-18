import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
def _register_methods(self, registry, instance_type):
    """
        Register method implementations.
        This simply registers that the method names are valid methods.  Inside
        of imp() below we retrieve the actual method to run from the type of
        the receiver argument (i.e. self).
        """
    to_register = list(instance_type.jit_methods) + list(instance_type.jit_static_methods)
    for meth in to_register:
        if meth not in self.implemented_methods:
            self._implement_method(registry, meth)
            self.implemented_methods.add(meth)