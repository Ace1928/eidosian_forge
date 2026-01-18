import unicodedata
import enchant.tokenize
def _consume_alpha_utf8(self, text, offset):
    """Consume a sequence of utf8 bytes forming an alphabetic character."""
    incr = 2
    u = ''
    while not u and incr <= 4:
        try:
            try:
                u = text[offset:offset + incr].decode('utf8')
            except AttributeError:
                try:
                    s = text[offset:offset + incr].tostring()
                except AttributeError:
                    s = ''.join([c for c in text[offset:offset + incr]])
                u = s.decode('utf8')
        except UnicodeDecodeError:
            incr += 1
    if not u:
        return 0
    if u.isalpha():
        return incr
    if unicodedata.category(u)[0] == 'M':
        return incr
    return 0