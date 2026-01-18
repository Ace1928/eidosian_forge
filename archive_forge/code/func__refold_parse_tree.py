import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def _refold_parse_tree(parse_tree, *, policy):
    """Return string of contents of parse_tree folded according to RFC rules.

    """
    maxlen = policy.max_line_length or sys.maxsize
    encoding = 'utf-8' if policy.utf8 else 'us-ascii'
    lines = ['']
    last_ew = None
    last_charset = None
    wrap_as_ew_blocked = 0
    want_encoding = False
    end_ew_not_allowed = Terminal('', 'wrap_as_ew_blocked')
    parts = list(parse_tree)
    while parts:
        part = parts.pop(0)
        if part is end_ew_not_allowed:
            wrap_as_ew_blocked -= 1
            continue
        tstr = str(part)
        if part.token_type == 'ptext' and set(tstr) & SPECIALS:
            want_encoding = True
        try:
            tstr.encode(encoding)
            charset = encoding
        except UnicodeEncodeError:
            if any((isinstance(x, errors.UndecodableBytesDefect) for x in part.all_defects)):
                charset = 'unknown-8bit'
            else:
                charset = 'utf-8'
            want_encoding = True
        if part.token_type == 'mime-parameters':
            _fold_mime_parameters(part, lines, maxlen, encoding)
            continue
        if want_encoding and (not wrap_as_ew_blocked):
            if not part.as_ew_allowed:
                want_encoding = False
                last_ew = None
                if part.syntactic_break:
                    encoded_part = part.fold(policy=policy)[:-len(policy.linesep)]
                    if policy.linesep not in encoded_part:
                        if len(encoded_part) > maxlen - len(lines[-1]):
                            newline = _steal_trailing_WSP_if_exists(lines)
                            lines.append(newline)
                        lines[-1] += encoded_part
                        continue
            if not hasattr(part, 'encode'):
                parts = list(part) + parts
            else:
                if last_ew is not None and charset != last_charset and (last_charset == 'unknown-8bit' or (last_charset == 'utf-8' and charset != 'us-ascii')):
                    last_ew = None
                last_ew = _fold_as_ew(tstr, lines, maxlen, last_ew, part.ew_combine_allowed, charset)
                last_charset = charset
            want_encoding = False
            continue
        if len(tstr) <= maxlen - len(lines[-1]):
            lines[-1] += tstr
            continue
        if part.syntactic_break and len(tstr) + 1 <= maxlen:
            newline = _steal_trailing_WSP_if_exists(lines)
            if newline or part.startswith_fws():
                lines.append(newline + tstr)
                last_ew = None
                continue
        if not hasattr(part, 'encode'):
            newparts = list(part)
            if not part.as_ew_allowed:
                wrap_as_ew_blocked += 1
                newparts.append(end_ew_not_allowed)
            parts = newparts + parts
            continue
        if part.as_ew_allowed and (not wrap_as_ew_blocked):
            parts.insert(0, part)
            want_encoding = True
            continue
        newline = _steal_trailing_WSP_if_exists(lines)
        if newline or part.startswith_fws():
            lines.append(newline + tstr)
        else:
            lines[-1] += tstr
    return policy.linesep.join(lines) + policy.linesep