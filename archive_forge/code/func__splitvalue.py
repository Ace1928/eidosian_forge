from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splitvalue(attr):
    """splitvalue('attr=value') --> 'attr', 'value'."""
    attr, delim, value = attr.partition('=')
    return (attr, value if delim else None)