import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def _fold(self, name, value, sanitize):
    parts = []
    parts.append('%s: ' % name)
    if isinstance(value, str):
        if _has_surrogates(value):
            if sanitize:
                h = header.Header(value, charset=_charset.UNKNOWN8BIT, header_name=name)
            else:
                parts.append(value)
                h = None
        else:
            h = header.Header(value, header_name=name)
    else:
        h = value
    if h is not None:
        maxlinelen = 0
        if self.max_line_length is not None:
            maxlinelen = self.max_line_length
        parts.append(h.encode(linesep=self.linesep, maxlinelen=maxlinelen))
    parts.append(self.linesep)
    return ''.join(parts)