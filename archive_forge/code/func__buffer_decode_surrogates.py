import re
import codecs
from typing import Tuple
from encodings.utf_8 import (
@staticmethod
def _buffer_decode_surrogates(sup, input, errors, final):
    """
        When we have improperly encoded surrogates, we can still see the
        bits that they were meant to represent.

        The surrogates were meant to encode a 20-bit number, to which we
        add 0x10000 to get a codepoint. That 20-bit number now appears in
        this form:

          11101101 1010abcd 10efghij 11101101 1011klmn 10opqrst

        The CESU8_RE above matches byte sequences of this form. Then we need
        to extract the bits and assemble a codepoint number from them.
        """
    if len(input) < 6:
        if final:
            return sup(input, errors, final)
        else:
            return ('', 0)
    elif CESU8_RE.match(input):
        codepoint = ((input[1] & 15) << 16) + ((input[2] & 63) << 10) + ((input[4] & 15) << 6) + (input[5] & 63) + 65536
        return (chr(codepoint), 6)
    else:
        return sup(input[:3], errors, False)