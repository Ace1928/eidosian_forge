import re
from json import scanner
def JSONObject(s_and_end, strict, scan_once, object_hook, object_pairs_hook, memo=None, _w=WHITESPACE.match, _ws=WHITESPACE_STR):
    s, end = s_and_end
    pairs = []
    pairs_append = pairs.append
    if memo is None:
        memo = {}
    memo_get = memo.setdefault
    nextchar = s[end:end + 1]
    if nextchar != '"':
        if nextchar in _ws:
            end = _w(s, end).end()
            nextchar = s[end:end + 1]
        if nextchar == '}':
            if object_pairs_hook is not None:
                result = object_pairs_hook(pairs)
                return (result, end + 1)
            pairs = {}
            if object_hook is not None:
                pairs = object_hook(pairs)
            return (pairs, end + 1)
        elif nextchar != '"':
            raise JSONDecodeError('Expecting property name enclosed in double quotes', s, end)
    end += 1
    while True:
        key, end = scanstring(s, end, strict)
        key = memo_get(key, key)
        if s[end:end + 1] != ':':
            end = _w(s, end).end()
            if s[end:end + 1] != ':':
                raise JSONDecodeError("Expecting ':' delimiter", s, end)
        end += 1
        try:
            if s[end] in _ws:
                end += 1
                if s[end] in _ws:
                    end = _w(s, end + 1).end()
        except IndexError:
            pass
        try:
            value, end = scan_once(s, end)
        except StopIteration as err:
            raise JSONDecodeError('Expecting value', s, err.value) from None
        pairs_append((key, value))
        try:
            nextchar = s[end]
            if nextchar in _ws:
                end = _w(s, end + 1).end()
                nextchar = s[end]
        except IndexError:
            nextchar = ''
        end += 1
        if nextchar == '}':
            break
        elif nextchar != ',':
            raise JSONDecodeError("Expecting ',' delimiter", s, end - 1)
        end = _w(s, end).end()
        nextchar = s[end:end + 1]
        end += 1
        if nextchar != '"':
            raise JSONDecodeError('Expecting property name enclosed in double quotes', s, end - 1)
    if object_pairs_hook is not None:
        result = object_pairs_hook(pairs)
        return (result, end)
    pairs = dict(pairs)
    if object_hook is not None:
        pairs = object_hook(pairs)
    return (pairs, end)