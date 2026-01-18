from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class ParseResultBytes(_ParseResultBase, _NetlocResultMixinBytes):
    __slots__ = ()

    def geturl(self):
        return urlunparse(self)