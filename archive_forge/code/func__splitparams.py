from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitparams(url):
    if '/' in url:
        i = url.find(';', url.rfind('/'))
        if i < 0:
            return (url, '')
    else:
        i = url.find(';')
    return (url[:i], url[i + 1:])