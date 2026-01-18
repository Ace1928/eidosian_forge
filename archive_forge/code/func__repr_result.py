import abc
import collections
import itertools
import weakref
from heat.common import exception
from heat.common.i18n import _
def _repr_result(self):
    return repr(self.parsed)