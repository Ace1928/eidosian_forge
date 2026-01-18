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
class ActiveComponent(Component):
    """A Component that makes semantic sense to activate or deactivate
    in a model.

    Private class attributes:
        _active         A boolean that is true if this component will be
                            used in model operations
    """

    def __init__(self, **kwds):
        self._active = True
        super(ActiveComponent, self).__init__(**kwds)

    @property
    def active(self):
        """Return the active attribute"""
        return self._active

    @active.setter
    def active(self, value):
        """Set the active attribute to the given value"""
        raise AttributeError('Assignment not allowed. Use the (de)activate methods.')

    def activate(self):
        """Set the active attribute to True"""
        self._active = True

    def deactivate(self):
        """Set the active attribute to False"""
        self._active = False