import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def _check_size_limit(self):
    if self._size_limit is not None:
        while len(self) > self._size_limit:
            self.popitem(last=False)