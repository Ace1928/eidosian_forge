from freetype.ft_types import *
class FT_Var_Axis(Structure):
    """
    A structure to model a given axis in design space for Multiple Masters,
    TrueType GX, and OpenType variation fonts.
    """
    _fields_ = [('name', FT_String_p), ('minimum', FT_Fixed), ('default', FT_Fixed), ('maximum', FT_Fixed), ('tag', FT_ULong), ('strid', FT_UInt)]