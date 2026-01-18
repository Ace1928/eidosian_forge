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
def _parse_encoding(tokens, _data):
    encoding = {}
    while True:
        token = next((t for t in tokens if t.is_keyword('StandardEncoding', 'dup', 'def')))
        if token.is_keyword('StandardEncoding'):
            return (_StandardEncoding, token.endpos())
        if token.is_keyword('def'):
            return (encoding, token.endpos())
        index_token = next(tokens)
        if not index_token.is_number():
            _log.warning(f'Parsing encoding: expected number, got {index_token}')
            continue
        name_token = next(tokens)
        if not name_token.is_slash_name():
            _log.warning(f'Parsing encoding: expected slash-name, got {name_token}')
            continue
        encoding[index_token.value()] = name_token.value()