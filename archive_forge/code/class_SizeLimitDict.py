import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
class SizeLimitDict(collections.OrderedDict):

    def __init__(self, *args, **kwargs):
        self._size_limit = kwargs.pop('size_limit', None)
        super(SizeLimitDict, self).__init__(*args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value):
        super(SizeLimitDict, self).__setitem__(key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self._size_limit is not None:
            while len(self) > self._size_limit:
                self.popitem(last=False)