from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class DefragResult(_DefragResultBase, _ResultMixinStr):
    __slots__ = ()

    def geturl(self):
        if self.fragment:
            return self.url + '#' + self.fragment
        else:
            return self.url