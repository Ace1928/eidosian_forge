import re as _re
from html.entities import html5 as _html5
def _replace_charref(s):
    s = s.group(1)
    if s[0] == '#':
        if s[1] in 'xX':
            num = int(s[2:].rstrip(';'), 16)
        else:
            num = int(s[1:].rstrip(';'))
        if num in _invalid_charrefs:
            return _invalid_charrefs[num]
        if 55296 <= num <= 57343 or num > 1114111:
            return '�'
        if num in _invalid_codepoints:
            return ''
        return chr(num)
    else:
        if s in _html5:
            return _html5[s]
        for x in range(len(s) - 1, 1, -1):
            if s[:x] in _html5:
                return _html5[s[:x]] + s[x:]
        else:
            return '&' + s