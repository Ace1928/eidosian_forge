import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def _sanitize_header(self, name, value):
    if not isinstance(value, str):
        return value
    if _has_surrogates(value):
        return header.Header(value, charset=_charset.UNKNOWN8BIT, header_name=name)
    else:
        return value