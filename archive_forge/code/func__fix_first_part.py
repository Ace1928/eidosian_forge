import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
@staticmethod
def _fix_first_part(part, boundary_marker):
    bm_len = len(boundary_marker)
    if boundary_marker == part[:bm_len]:
        return part[bm_len:]
    else:
        return part