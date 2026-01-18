import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.dae import ContinuousSet
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.base.indexed_component import UnindexedComponent_set, normalize_index
from pyomo.dae.flatten import (
class _TestFlattenBase(object):
    """A base class to hold the common _hashRef utility method.
    We don't just derive from Test... classes directly as this
    causes tests to run twice.

    """

    def _hashRef(self, ref):
        if not ref.is_indexed():
            return (id(ref),)
        else:
            return tuple(sorted((id(_) for _ in ref.values())))