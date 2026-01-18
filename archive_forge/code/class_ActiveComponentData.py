import logging
import sys
from copy import deepcopy
from pickle import PickleError
from weakref import ref as weakref_ref
import pyomo.common
from pyomo.common import DeveloperError
from pyomo.common.autoslots import AutoSlots, fast_deepcopy
from pyomo.common.collections import OrderedDict
from pyomo.common.deprecation import (
from pyomo.common.factory import Factory
from pyomo.common.formatting import tabular_writer, StreamIndenter
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base.component_namer import name_repr, index_repr
from pyomo.core.base.global_set import UnindexedComponent_index
class ActiveComponentData(ComponentData):
    """
    This is the base class for the component data used
    in Pyomo modeling components that can be activated and
    deactivated.

    It's possible to end up in a state where the parent Component
    has _active=True but all ComponentData have _active=False. This
    seems like a reasonable state, though we cannot easily detect
    this situation.  The important thing to avoid is the situation
    where one or more ComponentData are active, but the parent
    Component claims active=False. This class structure is designed
    to prevent this situation.

    Constructor arguments:
        owner           The component that owns this data object

    Private class attributes:
        _component      A weakref to the component that owns this data object
        _index          The index of this data object
        _active         A boolean that indicates whether this data is active
    """
    __slots__ = ('_active',)

    def __init__(self, component):
        super(ActiveComponentData, self).__init__(component)
        self._active = True

    @property
    def active(self):
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to a specified value."""
        raise AttributeError('Assignment not allowed. Use the (de)activate method')

    def activate(self):
        """Set the active attribute to True"""
        self._active = self.parent_component()._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False