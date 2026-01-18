from functools import wraps, partial
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.decorators import njit
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.typing.typeof import typeof_impl
from numba.experimental.jitclass import _box
def _generate_property(field, template, fname):
    """
    Generate simple function that get/set a field of the instance
    """
    source = template.format(field)
    glbls = {}
    exec(source, glbls)
    return njit(glbls[fname])