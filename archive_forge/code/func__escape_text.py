from pygments.formatter import Formatter
from pygments.util import get_int_opt, _surrogatepair
def _escape_text(self, text):
    if not text:
        return u''
    text = self._escape(text)
    buf = []
    for c in text:
        cn = ord(c)
        if cn < 2 ** 7:
            buf.append(str(c))
        elif 2 ** 7 <= cn < 2 ** 16:
            buf.append(u'{\\u%d}' % cn)
        elif 2 ** 16 <= cn:
            buf.append(u'{\\u%d}{\\u%d}' % _surrogatepair(cn))
    return u''.join(buf).replace(u'\n', u'\\par\n')