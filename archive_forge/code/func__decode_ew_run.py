from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import int, range, str, super, list
import re
from collections import namedtuple, OrderedDict
from future.backports.urllib.parse import (unquote, unquote_to_bytes)
from future.backports.email import _encoded_words as _ew
from future.backports.email import errors
from future.backports.email import utils
from the input.  Thus a parser method consumes the next syntactic construct
def _decode_ew_run(value):
    """ Decode a run of RFC2047 encoded words.

        _decode_ew_run(value) -> (text, value, defects)

    Scans the supplied value for a run of tokens that look like they are RFC
    2047 encoded words, decodes those words into text according to RFC 2047
    rules (whitespace between encoded words is discarded), and returns the text
    and the remaining value (including any leading whitespace on the remaining
    value), as well as a list of any defects encountered while decoding.  The
    input value may not have any leading whitespace.

    """
    res = []
    defects = []
    last_ws = ''
    while value:
        try:
            tok, ws, value = _wsp_splitter(value, 1)
        except ValueError:
            tok, ws, value = (value, '', '')
        if not (tok.startswith('=?') and tok.endswith('?=')):
            return (''.join(res), last_ws + tok + ws + value, defects)
        text, charset, lang, new_defects = _ew.decode(tok)
        res.append(text)
        defects.extend(new_defects)
        last_ws = ws
    return (''.join(res), last_ws, defects)