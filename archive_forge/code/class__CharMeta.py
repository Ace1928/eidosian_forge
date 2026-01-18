from __future__ import print_function, absolute_import
import sys
import struct
import os
from shibokensupport.signature import typing
from shibokensupport.signature.typing import TypeVar, Generic
from shibokensupport.signature.lib.tool import with_metaclass
class _CharMeta(type):

    def __repr__(self):
        return '%s.%s' % (self.__module__, self.__name__)