from __future__ import unicode_literals
from itertools import tee, chain
import re
import copy
class EndOfList(object):
    """Result of accessing element "-" of a list"""

    def __init__(self, list_):
        self.list_ = list_

    def __repr__(self):
        return '{cls}({lst})'.format(cls=self.__class__.__name__, lst=repr(self.list_))