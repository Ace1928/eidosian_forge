import collections
from requests import compat
def _format_header(name, value):
    return _coerce_to_bytes(name) + b': ' + _coerce_to_bytes(value) + b'\r\n'