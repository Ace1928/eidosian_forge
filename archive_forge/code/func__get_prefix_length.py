import struct
import sys
def _get_prefix_length(number1, number2, bits):
    """Get the number of leading bits that are same for two numbers.

    Args:
        number1: an integer.
        number2: another integer.
        bits: the maximum number of bits to compare.

    Returns:
        The number of leading bits that are the same for two numbers.

    """
    for i in range(bits):
        if number1 >> i == number2 >> i:
            return bits - i
    return 0