from sys import version_info as _swig_python_version_info
import weakref
class IntVarLocalSearchOperator(LocalSearchOperator):
    """
    Specialization of LocalSearchOperator built from an array of IntVars
    which specifies the scope of the operator.
    This class also takes care of storing current variable values in Start(),
    keeps track of changes done by the operator and builds the delta.
    The Deactivate() method can be used to perform Large Neighborhood Search.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, vars, keep_inverse_values=False):
        if self.__class__ == IntVarLocalSearchOperator:
            _self = None
        else:
            _self = self
        _pywrapcp.IntVarLocalSearchOperator_swiginit(self, _pywrapcp.new_IntVarLocalSearchOperator(_self, vars, keep_inverse_values))
    __swig_destroy__ = _pywrapcp.delete_IntVarLocalSearchOperator

    def Start(self, assignment):
        """
        This method should not be overridden. Override OnStart() instead which is
        called before exiting this method.
        """
        return _pywrapcp.IntVarLocalSearchOperator_Start(self, assignment)

    def IsIncremental(self):
        return _pywrapcp.IntVarLocalSearchOperator_IsIncremental(self)

    def Size(self):
        return _pywrapcp.IntVarLocalSearchOperator_Size(self)

    def Value(self, index):
        """
        Returns the value in the current assignment of the variable of given
        index.
        """
        return _pywrapcp.IntVarLocalSearchOperator_Value(self, index)

    def Var(self, index):
        """ Returns the variable of given index."""
        return _pywrapcp.IntVarLocalSearchOperator_Var(self, index)

    def OldValue(self, index):
        return _pywrapcp.IntVarLocalSearchOperator_OldValue(self, index)

    def PrevValue(self, index):
        return _pywrapcp.IntVarLocalSearchOperator_PrevValue(self, index)

    def SetValue(self, index, value):
        return _pywrapcp.IntVarLocalSearchOperator_SetValue(self, index, value)

    def Activated(self, index):
        return _pywrapcp.IntVarLocalSearchOperator_Activated(self, index)

    def Activate(self, index):
        return _pywrapcp.IntVarLocalSearchOperator_Activate(self, index)

    def Deactivate(self, index):
        return _pywrapcp.IntVarLocalSearchOperator_Deactivate(self, index)

    def AddVars(self, vars):
        return _pywrapcp.IntVarLocalSearchOperator_AddVars(self, vars)

    def OnStart(self):
        """
        Called by Start() after synchronizing the operator with the current
        assignment. Should be overridden instead of Start() to avoid calling
        IntVarLocalSearchOperator::Start explicitly.
        """
        return _pywrapcp.IntVarLocalSearchOperator_OnStart(self)

    def NextNeighbor(self, delta, deltadelta):
        """
        OnStart() should really be protected, but then SWIG doesn't see it. So we
        make it public, but only subclasses should access to it (to override it).
        Redefines MakeNextNeighbor to export a simpler interface. The calls to
        ApplyChanges() and RevertChanges() are factored in this method, hiding
        both delta and deltadelta from subclasses which only need to override
        MakeOneNeighbor().
        Therefore this method should not be overridden. Override MakeOneNeighbor()
        instead.
        """
        return _pywrapcp.IntVarLocalSearchOperator_NextNeighbor(self, delta, deltadelta)

    def OneNeighbor(self):
        """
        Creates a new neighbor. It returns false when the neighborhood is
        completely explored.
        MakeNextNeighbor() in a subclass of IntVarLocalSearchOperator.
        """
        return _pywrapcp.IntVarLocalSearchOperator_OneNeighbor(self)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_IntVarLocalSearchOperator(self)
        return weakref.proxy(self)