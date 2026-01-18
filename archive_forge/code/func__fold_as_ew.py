import re
import sys
import urllib   # For urllib.parse.unquote
from string import hexdigits
from operator import itemgetter
from email import _encoded_words as _ew
from email import errors
from email import utils
def _fold_as_ew(to_encode, lines, maxlen, last_ew, ew_combine_allowed, charset):
    """Fold string to_encode into lines as encoded word, combining if allowed.
    Return the new value for last_ew, or None if ew_combine_allowed is False.

    If there is already an encoded word in the last line of lines (indicated by
    a non-None value for last_ew) and ew_combine_allowed is true, decode the
    existing ew, combine it with to_encode, and re-encode.  Otherwise, encode
    to_encode.  In either case, split to_encode as necessary so that the
    encoded segments fit within maxlen.

    """
    if last_ew is not None and ew_combine_allowed:
        to_encode = str(get_unstructured(lines[-1][last_ew:] + to_encode))
        lines[-1] = lines[-1][:last_ew]
    if to_encode[0] in WSP:
        leading_wsp = to_encode[0]
        to_encode = to_encode[1:]
        if len(lines[-1]) == maxlen:
            lines.append(_steal_trailing_WSP_if_exists(lines))
        lines[-1] += leading_wsp
    trailing_wsp = ''
    if to_encode[-1] in WSP:
        trailing_wsp = to_encode[-1]
        to_encode = to_encode[:-1]
    new_last_ew = len(lines[-1]) if last_ew is None else last_ew
    encode_as = 'utf-8' if charset == 'us-ascii' else charset
    chrome_len = len(encode_as) + 7
    if chrome_len + 1 >= maxlen:
        raise errors.HeaderParseError('max_line_length is too small to fit an encoded word')
    while to_encode:
        remaining_space = maxlen - len(lines[-1])
        text_space = remaining_space - chrome_len
        if text_space <= 0:
            lines.append(' ')
            continue
        to_encode_word = to_encode[:text_space]
        encoded_word = _ew.encode(to_encode_word, charset=encode_as)
        excess = len(encoded_word) - remaining_space
        while excess > 0:
            to_encode_word = to_encode_word[:-1]
            encoded_word = _ew.encode(to_encode_word, charset=encode_as)
            excess = len(encoded_word) - remaining_space
        lines[-1] += encoded_word
        to_encode = to_encode[len(to_encode_word):]
        if to_encode:
            lines.append(' ')
            new_last_ew = len(lines[-1])
    lines[-1] += trailing_wsp
    return new_last_ew if ew_combine_allowed else None