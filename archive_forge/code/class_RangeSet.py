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
@ModelComponentFactory.register("A sequence of numeric values.  RangeSet(start,end,step) is a sequence starting a value 'start', and increasing in values by 'step' until a value greater than or equal to 'end' is reached.")
class RangeSet(Component):
    """A set object that represents a set of numeric values

    `RangeSet` objects are based around `NumericRange` objects, which
    include support for non-finite ranges (both continuous and
    unbounded). Similarly, boutique ranges (like semi-continuous
    domains) can be represented, e.g.:

    .. doctest::

       >>> from pyomo.core.base.range import NumericRange
       >>> from pyomo.environ import RangeSet
       >>> print(RangeSet(ranges=(NumericRange(0,0,0), NumericRange(1,100,0))))
       ([0] | [1..100])

    The `RangeSet` object continues to support the notation for
    specifying discrete ranges using "[first=1], last, [step=1]" values:

    .. doctest::

        >>> r = RangeSet(3)
        >>> print(r)
        [1:3]
        >>> print(list(r))
        [1, 2, 3]

        >>> r = RangeSet(2, 5)
        >>> print(r)
        [2:5]
        >>> print(list(r))
        [2, 3, 4, 5]

        >>> r = RangeSet(2, 5, 2)
        >>> print(r)
        [2:4:2]
        >>> print(list(r))
        [2, 4]

        >>> r = RangeSet(2.5, 4, 0.5)
        >>> print(r)
        ([2.5] | [3.0] | [3.5] | [4.0])
        >>> print(list(r))
        [2.5, 3.0, 3.5, 4.0]

    By implementing RangeSet using NumericRanges, the global Sets (like
    `Reals`, `Integers`, `PositiveReals`, etc.) are trivial
    instances of a RangeSet and support all Set operations.

    Parameters
    ----------
    *args: int | float | None
        The range defined by ([start=1], end, [step=1]).  If only a
        single positional parameter, `end` is supplied, then the
        RangeSet will be the integers starting at 1 up through and
        including end.  Providing two positional arguments, `x` and `y`,
        will result in a range starting at x up to and including y,
        incrementing by 1.  Providing a 3-tuple enables the
        specification of a step other than 1.

    finite: bool, optional
        This sets if this range is finite (discrete and bounded) or infinite

    ranges: iterable, optional
        The list of range objects that compose this RangeSet

    bounds: tuple, optional
        The lower and upper bounds of values that are admissible in this
        RangeSet

    filter: function, optional
        Function (rule) that returns True if the specified value is in
        the RangeSet or False if it is not.

    validate: function, optional
        Data validation function (rule).  The function will be called
        for every data member of the set, and if it returns False, a
        ValueError will be raised.

    name: str, optional
        Name for this component.

    doc: str, optional
        Text describing this component.
    """

    def __new__(cls, *args, **kwds):
        if cls is not RangeSet:
            return super(RangeSet, cls).__new__(cls)
        finite = kwds.pop('finite', None)
        if finite is None:
            if 'ranges' in kwds:
                if any((not r.isfinite() for r in kwds['ranges'])):
                    finite = False
            for i, _ in enumerate(args):
                if type(_) not in native_types:
                    if not isinstance(_, ComponentData) or not _.parent_component().is_constructed():
                        continue
                    else:
                        _ = value(_)
                if i < 2:
                    if _ in {None, _inf, -_inf}:
                        finite = False
                        break
                elif _ == 0 and args[0] is not args[1]:
                    finite = False
            if finite is None:
                finite = True
        if finite:
            return super(RangeSet, cls).__new__(AbstractFiniteScalarRangeSet)
        else:
            return super(RangeSet, cls).__new__(AbstractInfiniteScalarRangeSet)

    @overload
    def __init__(self, _end, *, finite=None, ranges=(), bounds=None, filter=None, validate=None, name=None, doc=None):
        ...

    @overload
    def __init__(self, _start, _end, _step=1, *, finite=None, ranges=(), bounds=None, filter=None, validate=None, name=None, doc=None):
        ...

    @overload
    def __init__(self, *, finite=None, ranges=(), bounds=None, filter=None, validate=None, name=None, doc=None):
        ...

    def __init__(self, *args, **kwds):
        kwds.setdefault('ctype', RangeSet)
        if len(args) > 3:
            raise ValueError('RangeSet expects 3 or fewer positional arguments (received %s)' % (len(args),))
        kwds.pop('finite', None)
        self._init_data = (args, kwds.pop('ranges', ()))
        self._init_validate = Initializer(kwds.pop('validate', None))
        self._init_filter = Initializer(kwds.pop('filter', None))
        self._init_bounds = kwds.pop('bounds', None)
        if self._init_bounds is not None:
            self._init_bounds = BoundsInitializer(self._init_bounds)
        Component.__init__(self, **kwds)
        try:
            if all((type(_) in native_types or (_.parent_component().is_constructed() and is_constant(_)) for _ in args)):
                self.construct()
        except AttributeError:
            pass

    def __str__(self):
        if self._name is not None:
            return self.name
        if not self._constructed:
            return type(self).__name__
        ans = ' | '.join((str(_) for _ in self.ranges()))
        if ' | ' in ans:
            return '(' + ans + ')'
        if ans:
            return ans
        else:
            return '[]'

    def construct(self, data=None):
        if self._constructed:
            return
        timer = ConstructionTimer(self)
        if is_debug_set(logger):
            logger.debug('Constructing RangeSet, name=%s, from data=%r' % (self, data))
        self._constructed = True
        if data is not None:
            raise ValueError('RangeSet.construct() does not support the data= argument.\nInitialization data (range endpoints) can only be supplied as numbers, constants, or Params to the RangeSet() declaration')
        args, ranges = self._init_data
        nonconstant_data_warning = any((not is_constant(arg) for arg in args))
        args = tuple((value(arg) for arg in args))
        if type(ranges) is not tuple:
            ranges = tuple(ranges)
        if len(args) == 1:
            if args[0] != 0:
                ranges = ranges + (NumericRange(1, args[0], 1),)
        elif len(args) == 2:
            if None in args or args[1] - args[0] != -1:
                args = (args[0], args[1], 1)
        if len(args) == 3:
            start, end, step = args
            if step:
                if start is None:
                    start, end = (end, start)
                    step *= -1
                if start is None:
                    ranges += (NumericRange(0, None, step), NumericRange(0, None, -step))
                elif int(step) != step:
                    if end is None:
                        raise ValueError('RangeSet does not support unbounded ranges with a non-integer step (got [%s:%s:%s])' % (start, end, step))
                    if (end >= start) ^ (step > 0):
                        raise ValueError('RangeSet: start, end ordering incompatible with step direction (got [%s:%s:%s])' % (start, end, step))
                    n = start
                    i = 0
                    while step > 0 and n <= end or (step < 0 and n >= end):
                        ranges += (NumericRange(n, n, 0),)
                        i += 1
                        n = start + step * i
                else:
                    ranges += (NumericRange(start, end, step),)
            else:
                ranges += (NumericRange(*args),)
        for r in ranges:
            if not isinstance(r, NumericRange):
                raise TypeError("RangeSet 'ranges' argument must be an iterable of NumericRange objects")
            if not r.isfinite() and self.isfinite():
                raise ValueError("Constructing a finite RangeSet over a non-finite range (%s).  Either correct the range data or specify 'finite=False' when declaring the RangeSet" % (r,))
        _block = self.parent_block()
        if self._init_bounds is not None:
            bnds = self._init_bounds(_block, None)
            tmp = []
            for r in ranges:
                tmp.extend(r.range_intersection(bnds.ranges()))
            ranges = tuple(tmp)
        self._ranges = ranges
        if self._init_filter is not None:
            if not self.isfinite():
                raise ValueError("The 'filter' keyword argument is not valid for non-finite RangeSet component (%s)" % (self.name,))
            try:
                _filter = Initializer(self._init_filter(_block, None))
                if _filter.constant():
                    _filter = self._init_filter
            except:
                _filter = self._init_filter
            new_ranges = []
            old_ranges = list(self.ranges())
            old_ranges.reverse()
            while old_ranges:
                r = old_ranges.pop()
                for i, val in enumerate(_FiniteRangeSetData._range_gen(r)):
                    if not _filter(_block, val):
                        split_r = r.range_difference((NumericRange(val, val, 0),))
                        if len(split_r) == 2:
                            new_ranges.append(split_r[0])
                            old_ranges.append(split_r[1])
                        elif len(split_r) == 1:
                            if i == 0:
                                old_ranges.append(split_r[0])
                            else:
                                new_ranges.append(split_r[0])
                        i = None
                        break
                if i is not None:
                    new_ranges.append(r)
            self._ranges = new_ranges
        if self._init_validate is not None:
            if not self.isfinite():
                raise ValueError("The 'validate' keyword argument is not valid for non-finite RangeSet component (%s)" % (self.name,))
            try:
                _validate = Initializer(self._init_validate(_block, None))
                if _validate.constant():
                    _validate = self._init_validate
            except:
                _validate = self._init_validate
            for val in self:
                try:
                    flag = _validate(_block, val)
                except:
                    logger.error("Exception raised while validating element '%s' for Set %s" % (val, self.name))
                    raise
                if not flag:
                    raise ValueError('The value=%s violates the validation rule of Set %s' % (val, self.name))
        if nonconstant_data_warning:
            logger.warning("Constructing RangeSet '%s' from non-constant data (e.g., Var or mutable Param).  The linkage between this RangeSet and the original source data will be broken, so updating the data value in the future will not be reflected in this RangeSet.  To suppress this warning, explicitly convert the source data to a constant type (e.g., float, int, or immutable Param)" % (self,))
        timer.report()

    def dim(self):
        return 0

    def index_set(self):
        return UnindexedComponent_set

    def _pprint(self):
        """
        Return data that will be printed for this component.
        """
        return ([('Dimen', self.dimen), ('Size', len(self) if self.isfinite() else 'Inf'), ('Bounds', self.bounds())], {None: self}.items(), ('Finite', 'Members'), lambda k, v: [v.isfinite(), ', '.join((str(r) for r in self.ranges())) or '[]'])