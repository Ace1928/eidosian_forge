import glob
import os
from io import StringIO
def _AsciiBase85Test(text='What is the average velocity of a sparrow?'):
    """Do the obvious test for whether Base 85 encoding works"""
    print('Plain text:', text)
    encoded = _AsciiBase85Encode(text)
    print('Encoded:', encoded)
    decoded = _AsciiBase85Decode(encoded)
    print('Decoded:', decoded)
    if decoded == text:
        print('Passed')
    else:
        print('Failed!')