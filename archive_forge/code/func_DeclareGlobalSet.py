import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
def DeclareGlobalSet(obj, caller_globals=None):
    """Declare a copy of a set as a global set in the calling module

    This takes a Set object and declares a duplicate of it as a
    GlobalSet object in the global namespace of the caller's module
    using the local name of the passed set.  GlobalSet objects are
    pseudo-singletons, in that copy.deepcopy (and Model.clone()) will
    not duplcicate them, and when you pickle and restore objects
    containing GlobalSets will still refer to the same object.  The
    declared GlobalSet object will be an instance of the original Set
    type.

    """
    obj.construct()
    assert obj.parent_component() is obj
    assert obj.parent_block() is None
    _name = obj.local_name
    if _name in GlobalSets and obj is not GlobalSets[_name]:
        raise RuntimeError('Duplicate Global Set declaration, %s' % (_name,))
    if caller_globals is None:
        caller_globals = inspect.currentframe().f_back.f_globals
    if _name in caller_globals and obj is not caller_globals[_name]:
        raise RuntimeError('Refusing to overwrite global object, %s' % (_name,))
    if _name in GlobalSets:
        _set = caller_globals[_name] = GlobalSets[_name]
        return _set

    class GlobalSet(GlobalSetBase, obj.__class__):
        __doc__ = '%s\n\n        References to this object will not be duplicated by deepcopy\n        and be maintained/restored by pickle.\n\n        ' % (obj.doc,)
        __slots__ = ('_bounds', '_interval')
        global_name = None

        def __new__(cls, *args, **kwds):
            """Hijack __new__ to mock up old RealSet el al. interface

            In the original Set implementation (Pyomo<=5.6.7), the
            global sets were instances of their own virtual set classes
            (RealSet, IntegerSet, BooleanSet), and one could create new
            instances of those sets with modified bounds.  Since the
            GlobalSet mechanism also declares new classes for every
            GlobalSet, we can mock up the old behavior through how we
            handle __new__().
            """
            if cls is GlobalSet and GlobalSet.global_name and issubclass(GlobalSet, RangeSet):
                deprecation_warning('The use of RealSet, IntegerSet, BinarySet and BooleanSet as Pyomo Set class generators is deprecated.  Please either use one of the pre-declared global Sets (e.g., Reals, NonNegativeReals, Integers, PositiveIntegers, Binary), or create a custom RangeSet.', version='5.7.1')
                base_set = GlobalSets[GlobalSet.global_name]
                bounds = kwds.pop('bounds', None)
                range_init = SetInitializer(base_set)
                if bounds is not None:
                    range_init.intersect(BoundsInitializer(bounds))
                name = name_kwd = kwds.pop('name', None)
                cls_name = kwds.pop('class_name', None)
                if name is None:
                    if cls_name is None:
                        name = base_set.name
                    else:
                        name = cls_name
                tmp = Set()
                ans = RangeSet(ranges=list(range_init(None, None, tmp).ranges()), name=name)
                ans._anonymous_sets = tmp._anonymous_sets
                if name_kwd is None and (cls_name is not None or bounds is not None):
                    ans._name += str(ans.bounds())
            else:
                ans = super(GlobalSet, cls).__new__(cls, *args, **kwds)
            if kwds:
                raise RuntimeError('Unexpected keyword arguments: %s' % (kwds,))
            return ans

        def bounds(self):
            return self._bounds

        def get_interval(self):
            return self._interval
    _set = GlobalSet()
    GlobalSets[_name] = caller_globals[_name] = _set
    GlobalSet.global_name = _name
    _set.__class__.__setstate__(_set, obj.__getstate__())
    _set._component = weakref.ref(_set)
    _set.construct()
    _set._bounds = obj.bounds()
    _set._interval = obj.get_interval()
    _set._anonymous_sets = GlobalSetBase
    return _set