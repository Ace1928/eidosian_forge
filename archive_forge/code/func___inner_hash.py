from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
@property
def __inner_hash(self):
    props = {'args': self.args, 'kwargs': self.kwargs}
    return get_hash(props)