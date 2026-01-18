import io
import struct
import typing
def _shift_bits(self, bits: typing.List[int], shifts: int) -> typing.List[int]:
    new_bits = [0] * 28
    for i in range(28):
        shift_index = i + shifts
        if shift_index >= 28:
            shift_index = shift_index - 28
        new_bits[i] = bits[shift_index]
    return new_bits