import asyncio
import functools
from typing import Tuple
class KeywordOnly(object):

    def double(self, *, count):
        return count * 2

    def triple(self, *, count):
        return count * 3

    def with_default(self, *, x='x'):
        print('x: ' + x)