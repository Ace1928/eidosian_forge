import array
import contextlib
import enum
import struct
@staticmethod
def ToTypedVector(element_type, fixed_len=0):
    """Converts element type to corresponding vector type.

    Args:
      element_type: vector element type
      fixed_len: number of elements: 0 for typed vector; 2, 3, or 4 for fixed
        typed vector.

    Returns:
      Typed vector type or fixed typed vector type.
    """
    if fixed_len == 0:
        if not Type.IsTypedVectorElementType(element_type):
            raise ValueError('must be typed vector element type')
    elif not Type.IsFixedTypedVectorElementType(element_type):
        raise ValueError('must be fixed typed vector element type')
    offset = element_type - Type.INT
    if fixed_len == 0:
        return Type(offset + Type.VECTOR_INT)
    elif fixed_len == 2:
        return Type(offset + Type.VECTOR_INT2)
    elif fixed_len == 3:
        return Type(offset + Type.VECTOR_INT3)
    elif fixed_len == 4:
        return Type(offset + Type.VECTOR_INT4)
    else:
        raise ValueError('unsupported fixed_len: %s' % fixed_len)