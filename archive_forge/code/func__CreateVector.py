import array
import contextlib
import enum
import struct
def _CreateVector(self, elements, typed, fixed, keys=None):
    """Writes vector elements to the underlying buffer."""
    length = len(elements)
    if fixed and (not typed):
        raise ValueError('fixed vector must be typed')
    bit_width = max(self._force_min_bit_width, BitWidth.U(length))
    prefix_elems = 1
    if keys:
        bit_width = max(bit_width, keys.ElemWidth(len(self._buf)))
        prefix_elems += 2
    vector_type = Type.KEY
    for i, e in enumerate(elements):
        bit_width = max(bit_width, e.ElemWidth(len(self._buf), prefix_elems + i))
        if typed:
            if i == 0:
                vector_type = e.Type
            elif vector_type != e.Type:
                raise RuntimeError('typed vector elements must be of the same type')
    if fixed and (not Type.IsFixedTypedVectorElementType(vector_type)):
        raise RuntimeError('must be fixed typed vector element type')
    byte_width = self._Align(bit_width)
    if keys:
        self._WriteOffset(keys.Value, byte_width)
        self._Write(U, 1 << keys.MinBitWidth, byte_width)
    if not fixed:
        self._Write(U, length, byte_width)
    loc = len(self._buf)
    for e in elements:
        self._WriteAny(e, byte_width)
    if not typed:
        for e in elements:
            self._buf.append(e.StoredPackedType(bit_width))
    if keys:
        type_ = Type.MAP
    elif typed:
        type_ = Type.ToTypedVector(vector_type, length if fixed else 0)
    else:
        type_ = Type.VECTOR
    return Value(loc, type_, bit_width)