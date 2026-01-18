from freetype.ft_types import *
class FT_Open_Args(Structure):
    """
    A structure used to indicate how to open a new font file or stream. A pointer
    to such a structure can be used as a parameter for the functions FT_Open_Face
    and FT_Attach_Stream.

    flags: A set of bit flags indicating how to use the structure.

    memory_base: The first byte of the file in memory.

    memory_size: The size in bytes of the file in memory.

    pathname: A pointer to an 8-bit file pathname.

    stream: A handle to a source stream object.

    driver: This field is exclusively used by FT_Open_Face; it simply specifies
            the font driver to use to open the face. If set to 0, FreeType
            tries to load the face with each one of the drivers in its list.

    num_params: The number of extra parameters.

    params: Extra parameters passed to the font driver when opening a new face.
    """
    _fields_ = [('flags', FT_UInt), ('memory_base', POINTER(FT_Byte)), ('memory_size', FT_Long), ('pathname', FT_String_p), ('stream', c_void_p), ('driver', c_void_p), ('num_params', FT_Int), ('params', FT_Parameter_p)]