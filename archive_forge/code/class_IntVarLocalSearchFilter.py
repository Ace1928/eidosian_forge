from sys import version_info as _swig_python_version_info
import weakref
class IntVarLocalSearchFilter(LocalSearchFilter):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, vars):
        if self.__class__ == IntVarLocalSearchFilter:
            _self = None
        else:
            _self = self
        _pywrapcp.IntVarLocalSearchFilter_swiginit(self, _pywrapcp.new_IntVarLocalSearchFilter(_self, vars))
    __swig_destroy__ = _pywrapcp.delete_IntVarLocalSearchFilter

    def Synchronize(self, assignment, delta):
        """
        This method should not be overridden. Override OnSynchronize() instead
        which is called before exiting this method.
        """
        return _pywrapcp.IntVarLocalSearchFilter_Synchronize(self, assignment, delta)

    def Size(self):
        return _pywrapcp.IntVarLocalSearchFilter_Size(self)

    def Value(self, index):
        return _pywrapcp.IntVarLocalSearchFilter_Value(self, index)

    def IndexFromVar(self, var):
        return _pywrapcp.IntVarLocalSearchFilter_IndexFromVar(self, var)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_IntVarLocalSearchFilter(self)
        return weakref.proxy(self)