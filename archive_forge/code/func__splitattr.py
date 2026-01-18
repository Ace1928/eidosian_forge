from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitattr(url):
    """splitattr('/path;attr1=value1;attr2=value2;...') ->
        '/path', ['attr1=value1', 'attr2=value2', ...]."""
    words = url.split(';')
    return (words[0], words[1:])