from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splittag(url):
    """splittag('/path#tag') --> '/path', 'tag'."""
    path, delim, tag = url.rpartition('#')
    if delim:
        return (path, tag)
    return (url, None)