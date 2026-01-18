from __future__ import print_function, absolute_import
import sys
import re
import warnings
import types
import keyword
import functools
from shibokensupport.signature.mapping import (type_map, update_mapping,
from shibokensupport.signature.lib.tool import (SimpleNamespace,
from inspect import currentframe
def _resolve_arraytype(thing, line):
    search = re.search('\\[(\\d*)\\]$', thing)
    thing = thing[:search.start()]
    if thing.endswith(']'):
        thing = _resolve_arraytype(thing, line)
    if search.group(1):
        nelem = int(search.group(1))
        thing = ', '.join([thing] * nelem)
        thing = 'Tuple[' + thing + ']'
    else:
        thing = 'QList[' + thing + ']'
    return thing