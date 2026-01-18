from abc import abstractmethod, abstractproperty
from typing import List, Optional, Tuple, Union
from parso.utils import split_lines
class ErrorLeaf(Leaf):
    """
    A leaf that is either completely invalid in a language (like `$` in Python)
    or is invalid at that position. Like the star in `1 +* 1`.
    """
    __slots__ = ('token_type',)
    type = 'error_leaf'

    def __init__(self, token_type, value, start_pos, prefix=''):
        super().__init__(value, start_pos, prefix)
        self.token_type = token_type

    def __repr__(self):
        return '<%s: %s:%s, %s>' % (type(self).__name__, self.token_type, repr(self.value), self.start_pos)