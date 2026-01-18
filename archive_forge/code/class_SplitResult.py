from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class SplitResult(_SplitResultBase, _NetlocResultMixinStr):
    __slots__ = ()

    def geturl(self):
        return urlunsplit(self)