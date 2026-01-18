from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class SplitResultBytes(_SplitResultBase, _NetlocResultMixinBytes):
    __slots__ = ()

    def geturl(self):
        return urlunsplit(self)