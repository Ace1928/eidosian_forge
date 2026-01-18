from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
class ordereddict(OrderedDict):
    if not hasattr(OrderedDict, 'insert'):

        def insert(self, pos, key, value):
            if pos >= len(self):
                self[key] = value
                return
            od = ordereddict()
            od.update(self)
            for k in od:
                del self[k]
            for index, old_key in enumerate(od):
                if pos == index:
                    self[key] = value
                self[old_key] = od[old_key]