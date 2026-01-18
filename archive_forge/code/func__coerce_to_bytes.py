import collections
from requests import compat
def _coerce_to_bytes(data):
    if not isinstance(data, bytes) and hasattr(data, 'encode'):
        data = data.encode('utf-8')
    return data if data is not None else b''