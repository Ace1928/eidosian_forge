import os
from ... import urlutils
from . import request
def _deserialise_optional_mode(mode):
    if mode == b'':
        return None
    else:
        return int(mode)