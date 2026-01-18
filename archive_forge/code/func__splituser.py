from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _splituser(host):
    """splituser('user[:passwd]@host[:port]') --> 'user[:passwd]', 'host[:port]'."""
    user, delim, host = host.rpartition('@')
    return (user if delim else None, host)