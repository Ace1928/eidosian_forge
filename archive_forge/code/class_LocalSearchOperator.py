from sys import version_info as _swig_python_version_info
import weakref
class LocalSearchOperator(BaseObject):
    """
    The base class for all local search operators.

    A local search operator is an object that defines the neighborhood of a
    solution. In other words, a neighborhood is the set of solutions which can
    be reached from a given solution using an operator.

    The behavior of the LocalSearchOperator class is similar to iterators.
    The operator is synchronized with an assignment (gives the
    current values of the variables); this is done in the Start() method.

    Then one can iterate over the neighbors using the MakeNextNeighbor method.
    This method returns an assignment which represents the incremental changes
    to the current solution. It also returns a second assignment representing
    the changes to the last solution defined by the neighborhood operator; this
    assignment is empty if the neighborhood operator cannot track this
    information.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self, *args, **kwargs):
        raise AttributeError('No constructor defined - class is abstract')
    __repr__ = _swig_repr

    def NextNeighbor(self, delta, deltadelta):
        return _pywrapcp.LocalSearchOperator_NextNeighbor(self, delta, deltadelta)

    def Start(self, assignment):
        return _pywrapcp.LocalSearchOperator_Start(self, assignment)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_LocalSearchOperator(self)
        return weakref.proxy(self)