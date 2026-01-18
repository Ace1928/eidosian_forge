from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
def __get_hash(self):
    hashes = self.__upstream_hashes + [self.__inner_hash]
    return get_hash_int(hashes)