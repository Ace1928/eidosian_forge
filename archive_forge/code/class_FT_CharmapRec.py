from freetype.ft_types import *
class FT_CharmapRec(Structure):
    """
    The base charmap structure.

    face : A handle to the parent face object.

    encoding : An FT_Encoding tag identifying the charmap. Use this with
               FT_Select_Charmap.

    platform_id: An ID number describing the platform for the following
                 encoding ID. This comes directly from the TrueType
                 specification and should be emulated for other formats.

    encoding_id: A platform specific encoding number. This also comes from the
                 TrueType specification and should be emulated similarly.
    """
    _fields_ = [('face', c_void_p), ('encoding', FT_Encoding), ('platform_id', FT_UShort), ('encoding_id', FT_UShort)]