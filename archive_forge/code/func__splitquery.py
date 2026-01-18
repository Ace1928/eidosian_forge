from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitquery(url):
    """splitquery('/path?query') --> '/path', 'query'."""
    path, delim, query = url.rpartition('?')
    if delim:
        return (path, query)
    return (url, None)