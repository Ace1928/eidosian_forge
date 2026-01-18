import unittest
import binascii
from Cryptodome.Util.py3compat import b
def a2b_hex(s):
    """Convert hexadecimal to binary, ignoring whitespace"""
    return binascii.a2b_hex(strip_whitespace(s))