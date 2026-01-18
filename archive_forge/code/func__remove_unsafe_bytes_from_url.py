import re
import sys
import collections
from collections import namedtuple
def _remove_unsafe_bytes_from_url(url):
    for b in _UNSAFE_URL_BYTES_TO_REMOVE:
        url = url.replace(b, '')
    return url