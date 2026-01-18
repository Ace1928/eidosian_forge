from sys import version_info as _swig_python_version_info
import weakref
class DecisionBuilder(BaseObject):
    """
    A DecisionBuilder is responsible for creating the search tree. The
    important method is Next(), which returns the next decision to execute.
    """
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')

    def __init__(self):
        if self.__class__ == DecisionBuilder:
            _self = None
        else:
            _self = self
        _pywrapcp.DecisionBuilder_swiginit(self, _pywrapcp.new_DecisionBuilder(_self))
    __swig_destroy__ = _pywrapcp.delete_DecisionBuilder

    def NextWrapper(self, s):
        """
        This is the main method of the decision builder class. It must
        return a decision (an instance of the class Decision). If it
        returns nullptr, this means that the decision builder has finished
        its work.
        """
        return _pywrapcp.DecisionBuilder_NextWrapper(self, s)

    def DebugString(self):
        return _pywrapcp.DecisionBuilder_DebugString(self)

    def __repr__(self):
        return _pywrapcp.DecisionBuilder___repr__(self)

    def __str__(self):
        return _pywrapcp.DecisionBuilder___str__(self)

    def __disown__(self):
        self.this.disown()
        _pywrapcp.disown_DecisionBuilder(self)
        return weakref.proxy(self)