import base64
import struct
from os_ken.lib import addrconv
def _split_str(s, n):
    """
    split string into list of strings by specified number.
    """
    length = len(s)
    return [s[i:i + n] for i in range(0, length, n)]