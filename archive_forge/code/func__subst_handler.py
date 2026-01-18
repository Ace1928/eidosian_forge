import re
import sys
from builtins import str, chr
def _subst_handler(matchobj):
    src = matchobj.group(0)
    hiv = ord(src[0])
    if hiv < 55296:
        return ' '
    return XLAT[65536 + ((hiv & 1023) << 10) | ord(src[1]) & 1023]