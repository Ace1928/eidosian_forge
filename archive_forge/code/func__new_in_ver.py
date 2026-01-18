import sys
from abc import ABCMeta
from collections import OrderedDict
from collections import UserDict
from collections import UserList
from collections import UserString
from collections import abc
from zope.interface.common import ABCInterface
from zope.interface.common import optional
def _new_in_ver(name, ver, bases_if_missing=(ABCMeta,), register_if_missing=()):
    if ver:
        return getattr(abc, name)
    missing = ABCMeta(name, bases_if_missing, {'__doc__': 'The ABC %s is not defined in this version of Python.' % name})
    for c in register_if_missing:
        missing.register(c)
    return missing