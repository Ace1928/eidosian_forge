from abc import get_cache_token
from collections import namedtuple
from reprlib import recursive_repr
from _thread import RLock
from types import GenericAlias
class K(object):
    __slots__ = ['obj']

    def __init__(self, obj):
        self.obj = obj

    def __lt__(self, other):
        return mycmp(self.obj, other.obj) < 0

    def __gt__(self, other):
        return mycmp(self.obj, other.obj) > 0

    def __eq__(self, other):
        return mycmp(self.obj, other.obj) == 0

    def __le__(self, other):
        return mycmp(self.obj, other.obj) <= 0

    def __ge__(self, other):
        return mycmp(self.obj, other.obj) >= 0
    __hash__ = None