import glob
import os
from io import StringIO
def _AsciiHexTest(text='What is the average velocity of a sparrow?'):
    """Do the obvious test for whether Ascii Hex encoding works"""
    print('Plain text:', text)
    encoded = _AsciiHexEncode(text)
    print('Encoded:', encoded)
    decoded = _AsciiHexDecode(encoded)
    print('Decoded:', decoded)
    if decoded == text:
        print('Passed')
    else:
        print('Failed!')