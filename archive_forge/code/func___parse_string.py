import re
import string
import types
def __parse_string(self, str, patt=_CookiePattern):
    i = 0
    n = len(str)
    parsed_items = []
    morsel_seen = False
    TYPE_ATTRIBUTE = 1
    TYPE_KEYVALUE = 2
    while 0 <= i < n:
        match = patt.match(str, i)
        if not match:
            break
        key, value = (match.group('key'), match.group('val'))
        i = match.end(0)
        if key[0] == '$':
            if not morsel_seen:
                continue
            parsed_items.append((TYPE_ATTRIBUTE, key[1:], value))
        elif key.lower() in Morsel._reserved:
            if not morsel_seen:
                return
            if value is None:
                if key.lower() in Morsel._flags:
                    parsed_items.append((TYPE_ATTRIBUTE, key, True))
                else:
                    return
            else:
                parsed_items.append((TYPE_ATTRIBUTE, key, _unquote(value)))
        elif value is not None:
            parsed_items.append((TYPE_KEYVALUE, key, self.value_decode(value)))
            morsel_seen = True
        else:
            return
    M = None
    for tp, key, value in parsed_items:
        if tp == TYPE_ATTRIBUTE:
            assert M is not None
            M[key] = value
        else:
            assert tp == TYPE_KEYVALUE
            rval, cval = value
            self.__set(key, rval, cval)
            M = self[key]