import functools
import os
from ..common.build import is_hip
from . import core
@core.extern
def acos(arg0, _builder=None):
    return core.extern_elementwise('libdevice', libdevice_path(), [arg0], {(core.dtype('fp32'),): ('__nv_acosf', core.dtype('fp32')), (core.dtype('fp64'),): ('__nv_acos', core.dtype('fp64'))}, is_pure=True, _builder=_builder)