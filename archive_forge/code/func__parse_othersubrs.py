import binascii
import functools
import logging
import re
import string
import struct
import numpy as np
from matplotlib.cbook import _format_approx
from . import _api
@staticmethod
def _parse_othersubrs(tokens, data):
    init_pos = None
    while True:
        token = next(tokens)
        if init_pos is None:
            init_pos = token.pos
        if token.is_delim():
            _expression(token, tokens, data)
        elif token.is_keyword('def', 'ND', '|-'):
            return (data[init_pos:token.endpos()], token.endpos())