import copy
import logging
import sys
import weakref
import textwrap
from collections import defaultdict
from contextlib import contextmanager
from inspect import isclass, currentframe
from io import StringIO
from itertools import filterfalse, chain
from operator import itemgetter, attrgetter
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import Mapping
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.formatting import StreamIndenter
from pyomo.common.gc_manager import PauseGC
from pyomo.common.log import is_debug_set
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import (
from pyomo.core.base.enums import SortComponents, TraversalStrategy
from pyomo.core.base.global_set import UnindexedComponent_index
from pyomo.core.base.componentuid import ComponentUID
from pyomo.core.base.set import Any
from pyomo.core.base.var import Var
from pyomo.core.base.initializer import Initializer
from pyomo.core.base.indexed_component import (
from pyomo.opt.base import ProblemFormat, guess_format
from pyomo.opt import WriterFactory
def _component_data_iteritems(self, ctype, active, sort, dedup):
    """return the name, index, and component data for matching ctypes

        Generator that returns a nested 2-tuple of

            ((component name, index value), _ComponentData)

        for every component data in the block matching the specified
        ctype(s).

        Parameters
        ----------
        ctype:  None or type or iterable
            Specifies the component types (`ctypes`) to include

        active: None or bool
            Filter components by the active flag

        sort: None or bool or SortComponents
            Iterate over the components in a specified sorted order

        dedup: _DeduplicateInfo
            Deduplicator to prevent returning the same _ComponentData twice
        """
    for name, comp in PseudoMap(self, ctype, active, sort).items():
        if comp.is_indexed():
            _items = comp.items(sort)
        elif hasattr(comp, '_data'):
            assert len(comp._data) <= 1
            _items = comp._data.items()
        else:
            _items = ((None, comp),)
        if active is None or not isinstance(comp, ActiveIndexedComponent):
            _items = (((name, idx), compData) for idx, compData in _items)
        else:
            _items = (((name, idx), compData) for idx, compData in _items if compData.active == active)
        yield from dedup.unique(comp, _items, False)