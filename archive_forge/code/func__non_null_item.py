import abc
import collections
import itertools
import weakref
from heat.common import exception
from heat.common.i18n import _
def _non_null_item(i):
    k, v = i
    return v is not Ellipsis