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
class PseudoMap(AutoSlots.Mixin):
    """
    This class presents a "mock" dict interface to the internal
    _BlockData data structures.  We return this object to the
    user to preserve the historical "{ctype : {name : obj}}"
    interface without actually regenerating that dict-of-dicts data
    structure.

    We now support {ctype : PseudoMap()}
    """
    __slots__ = ('_block', '_ctypes', '_active', '_sorted')

    def __init__(self, block, ctype, active=None, sort=False):
        """
        TODO
        """
        self._block = block
        if isclass(ctype):
            self._ctypes = {ctype}
        elif ctype is None:
            self._ctypes = Any
        elif ctype.__class__ is SubclassOf:
            self._ctypes = ctype
        else:
            self._ctypes = set(ctype)
        self._active = active
        self._sorted = SortComponents.ALPHABETICAL in SortComponents(sort)

    def __iter__(self):
        """
        TODO
        """
        return self.keys()

    def __getitem__(self, key):
        """
        TODO
        """
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if x[0].ctype in self._ctypes:
                if self._active is None or x[0].active == self._active:
                    return x[0]
        msg = ''
        if self._active is not None:
            msg += self._active and 'active ' or 'inactive '
        if self._ctypes is not Any:
            if len(self._ctypes) == 1:
                msg += next(iter(self._ctypes)).__name__ + ' '
            else:
                types = sorted((x.__name__ for x in self._ctypes))
                msg += '%s or %s ' % (', '.join(types[:-1]), types[-1])
        raise KeyError("%scomponent '%s' not found in block %s" % (msg, key, self._block.name))

    def __nonzero__(self):
        """
        TODO
        """
        sort_order = self._sorted
        try:
            self._sorted = False
            for x in self.values():
                return True
            return False
        finally:
            self._sorted = sort_order
    __bool__ = __nonzero__

    def __len__(self):
        """
        TODO
        """
        if self._active is None:
            if self._ctypes is Any:
                return sum((x[2] for x in self._block._ctypes.values()))
            else:
                return sum((self._block._ctypes[x][2] for x in self._block._ctypes if x in self._ctypes))
        ans = 0
        for x in self.values():
            ans += 1
        return ans

    def __contains__(self, key):
        """
        TODO
        """
        if key in self._block._decl:
            x = self._block._decl_order[self._block._decl[key]]
            if x[0].ctype in self._ctypes:
                return self._active is None or x[0].active == self._active
        return False

    def _ctypewalker(self):
        """
        TODO
        """
        _decl_order = self._block._decl_order
        if self._ctypes.__class__ is set:
            _idx_list = [self._block._ctypes[x][0] for x in self._ctypes if x in self._block._ctypes]
        else:
            _idx_list = [self._block._ctypes[x][0] for x in self._block._ctypes if x in self._ctypes]
        _idx_list.sort(reverse=True)
        while _idx_list:
            _idx = _idx_list.pop()
            _next_ctype = _idx_list[-1] if _idx_list else None
            while 1:
                _obj, _idx = _decl_order[_idx]
                if _obj is not None:
                    yield _obj
                if _idx is None:
                    break
                if _next_ctype is not None and _idx > _next_ctype:
                    _idx_list.append(_idx)
                    _idx_list.sort(reverse=True)
                    break

    def keys(self):
        """
        Generator returning the component names defined on the Block
        """
        return map(attrgetter('_name'), self.values())

    def values(self):
        """
        Generator returning the components defined on the Block
        """
        if self._ctypes is Any:
            walker = filter(_isNotNone, map(itemgetter(0), self._block._decl_order))
        else:
            walker = self._ctypewalker()
        if self._active:
            walker = filter(attrgetter('active'), walker)
        elif self._active is not None:
            walker = filterfalse(attrgetter('active'), walker)
        if self._sorted:
            return iter(sorted(walker, key=attrgetter('_name')))
        else:
            return walker

    def items(self):
        """
        Generator returning (name, component) tuples for components
        defined on the Block
        """
        for obj in self.values():
            yield (obj._name, obj)

    @deprecated('The iterkeys method is deprecated. Use dict.keys().', version='6.0')
    def iterkeys(self):
        """
        Generator returning the component names defined on the Block
        """
        return self.keys()

    @deprecated('The itervalues method is deprecated. Use dict.values().', version='6.0')
    def itervalues(self):
        """
        Generator returning the components defined on the Block
        """
        return self.values()

    @deprecated('The iteritems method is deprecated. Use dict.items().', version='6.0')
    def iteritems(self):
        """
        Generator returning (name, component) tuples for components
        defined on the Block
        """
        return self.items()