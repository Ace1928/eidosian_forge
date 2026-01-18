from sys import version_info as _swig_python_version_info
import weakref
class BoundCost(object):
    """
    A structure meant to store soft bounds and associated violation constants.
    It is 'Simple' because it has one BoundCost per element,
    in contrast to 'Multiple'. Design notes:
    - it is meant to store model information to be shared through pointers,
      so it disallows copy and assign to avoid accidental duplication.
    - it keeps soft bounds as an array of structs to help cache,
      because code that uses such bounds typically use both bound and cost.
    - soft bounds are named pairs, prevents some mistakes.
    - using operator[] to access elements is not interesting,
      because the structure will be accessed through pointers, moreover having
      to type bound_cost reminds the user of the order if they do a copy
      assignment of the element.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    bound = property(_pywrapcp.BoundCost_bound_get, _pywrapcp.BoundCost_bound_set)
    cost = property(_pywrapcp.BoundCost_cost_get, _pywrapcp.BoundCost_cost_set)

    def __init__(self, *args):
        _pywrapcp.BoundCost_swiginit(self, _pywrapcp.new_BoundCost(*args))
    __swig_destroy__ = _pywrapcp.delete_BoundCost