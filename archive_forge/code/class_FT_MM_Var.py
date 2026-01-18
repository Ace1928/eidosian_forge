from freetype.ft_types import *
class FT_MM_Var(Structure):
    """
    A structure to model the axes and space of an Adobe MM, TrueType GX,
    or OpenType variation font.
    Some fields are specific to one format and not to the others.
    """
    _fields_ = [('num_axis', FT_UInt), ('num_designs', FT_UInt), ('num_namedstyles', FT_UInt), ('axis', POINTER(FT_Var_Axis)), ('namedstyle', POINTER(FT_Var_Named_Style))]