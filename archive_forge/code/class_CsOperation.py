from ctypes import POINTER, c_byte, c_double, c_int, c_uint
from django.contrib.gis.geos.libgeos import CS_PTR, GEOM_PTR, GEOSFuncFactory
from django.contrib.gis.geos.prototypes.errcheck import GEOSException, last_arg_byref
class CsOperation(GEOSFuncFactory):
    """For coordinate sequence operations."""
    restype = c_int

    def __init__(self, *args, ordinate=False, get=False, **kwargs):
        if get:
            errcheck = check_cs_get
            dbl_param = POINTER(c_double)
        else:
            errcheck = check_cs_op
            dbl_param = c_double
        if ordinate:
            argtypes = [CS_PTR, c_uint, c_uint, dbl_param]
        else:
            argtypes = [CS_PTR, c_uint, dbl_param]
        super().__init__(*args, **{**kwargs, 'errcheck': errcheck, 'argtypes': argtypes})