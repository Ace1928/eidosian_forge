from collections import namedtuple, Counter
from os.path import commonprefix
def _common_shorten_repr(*args):
    args = tuple(map(safe_repr, args))
    maxlen = max(map(len, args))
    if maxlen <= _MAX_LENGTH:
        return args
    prefix = commonprefix(args)
    prefixlen = len(prefix)
    common_len = _MAX_LENGTH - (maxlen - prefixlen + _MIN_BEGIN_LEN + _PLACEHOLDER_LEN)
    if common_len > _MIN_COMMON_LEN:
        assert _MIN_BEGIN_LEN + _PLACEHOLDER_LEN + _MIN_COMMON_LEN + (maxlen - prefixlen) < _MAX_LENGTH
        prefix = _shorten(prefix, _MIN_BEGIN_LEN, common_len)
        return tuple((prefix + s[prefixlen:] for s in args))
    prefix = _shorten(prefix, _MIN_BEGIN_LEN, _MIN_COMMON_LEN)
    return tuple((prefix + _shorten(s[prefixlen:], _MIN_DIFF_LEN, _MIN_END_LEN) for s in args))