import re
import string
import sys
from jsbeautifier.unpackers import UnpackingError
def _dictunbaser(self, string):
    """Decodes a  value to an integer."""
    ret = 0
    for index, cipher in enumerate(string[::-1]):
        ret += self.base ** index * self.dictionary[cipher]
    return ret