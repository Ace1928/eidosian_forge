import abc
import collections
from collections import abc as cabc
import itertools
from oslo_utils import reflection
from taskflow.types import sets
from taskflow.utils import misc
def _build_arg_mapping(self, executor, requires=None, rebind=None, auto_extract=True, ignore_list=None):
    required, optional = _build_arg_mapping(self.name, requires, rebind, executor, auto_extract, ignore_list=ignore_list)
    rebind = collections.OrderedDict()
    for arg_name, bound_name in itertools.chain(required.items(), optional.items()):
        rebind.setdefault(arg_name, bound_name)
    requires = sets.OrderedSet(required.values())
    optional = sets.OrderedSet(optional.values())
    if self.inject:
        inject_keys = frozenset(self.inject.keys())
        requires -= inject_keys
        optional -= inject_keys
    return (rebind, requires, optional)