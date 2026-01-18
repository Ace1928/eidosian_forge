import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
def _itervalues(self):
    return (self[k] for k in self._iterkeys())