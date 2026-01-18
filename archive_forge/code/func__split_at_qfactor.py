def _split_at_qfactor(s):
    """Splits a string at the quality factor (;q=) parameter.

    Returns the left and right substrings as a two-member tuple.

    """
    pos = 0
    while 0 <= pos < len(s):
        pos = s.find(';', pos)
        if pos < 0:
            break
        startpos = pos
        pos = pos + 1
        while pos < len(s) and s[pos] in LWS:
            pos = pos + 1
        if pos < len(s) and s[pos] == 'q':
            pos = pos + 1
            while pos < len(s) and s[pos] in LWS:
                pos = pos + 1
            if pos < len(s) and s[pos] == '=':
                pos = pos + 1
                while pos < len(s) and s[pos] in LWS:
                    pos = pos + 1
                return (s[:startpos], s[pos:])
    return (s, '')