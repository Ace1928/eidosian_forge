from ctypes import *

Freetype basic data types
-------------------------

FT_Byte : A simple typedef for the unsigned char type.

FT_Bytes : A typedef for constant memory areas.

FT_Char : A simple typedef for the signed char type.

FT_Int : A typedef for the int type.

FT_UInt : A typedef for the unsigned int type.

FT_Int16 : A typedef for a 16bit signed integer type.

FT_UInt16 : A typedef for a 16bit unsigned integer type.

FT_Int32 : A typedef for a 32bit signed integer type.

FT_UInt32 : A typedef for a 32bit unsigned integer type.

FT_Short : A typedef for signed short.

FT_UShort : A typedef for unsigned short.

FT_Long : A typedef for signed long.

FT_ULong : A typedef for unsigned long.

FT_Bool : A typedef of unsigned char, used for simple booleans. As usual,
          values 1 and 0 represent true and false, respectively.

FT_Offset : This is equivalent to the ANSI C 'size_t' type, i.e., the largest
            unsigned integer type used to express a file size or position, or
            a memory block size.

FT_PtrDist : This is equivalent to the ANSI C 'ptrdiff_t' type, i.e., the
             largest signed integer type used to express the distance between
             two pointers.

FT_String : A simple typedef for the char type, usually used for strings. 

FT_Tag  : A typedef for 32-bit tags (as used in the SFNT format).

FT_Error : The FreeType error code type. A value of 0 is always interpreted as
           a successful operation.

FT_Fixed : This type is used to store 16.16 fixed float values, like scaling
           values or matrix coefficients.

FT_Pointer : A simple typedef for a typeless pointer.

FT_Pos : The type FT_Pos is used to store vectorial coordinates. Depending on
         the context, these can represent distances in integer font units, or
         16.16, or 26.6 fixed float pixel coordinates.

FT_FWord : A signed 16-bit integer used to store a distance in original font
           units.

FT_UFWord : An unsigned 16-bit integer used to store a distance in original
            font units.

FT_F2Dot14 : A signed 2.14 fixed float type used for unit vectors.

FT_F26Dot6 : A signed 26.6 fixed float type used for vectorial pixel
             coordinates.
