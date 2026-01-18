import ctypes
from OpenGL.constant import Constant
from OpenGL._bytes import bytes,unicode,as_8_bit, long
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
from OpenGL.platform import PLATFORM as _p
def _defineType(name, baseType, convertFunc=long):
    from OpenGL import _configflags
    do_wrapping = _configflags.ALLOW_NUMPY_SCALARS or (ctypes_version < [1, 1, 0] and baseType in (ctypes.c_uint, ctypes.c_uint64, ctypes.c_ulong, ctypes.c_ushort) or (ctypes_version < [1, 0, 2] and baseType in (ctypes.c_int, ctypes.c_int64, ctypes.c_long, ctypes.c_short)))
    if do_wrapping:
        original = baseType.from_param
        if not getattr(original, 'from_param_numpy_scalar', False):

            def from_param(x, typeCode=None):
                try:
                    return original(x)
                except TypeError as err:
                    try:
                        return original(convertFunc(x))
                    except TypeError:
                        raise err
            from_param = staticmethod(from_param)
            setattr(baseType, 'from_param', from_param)
            baseType.from_param_numpy_scalar = True
        return baseType
    else:
        return baseType