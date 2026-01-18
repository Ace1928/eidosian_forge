import ctypes, ctypes.util, operator, sys
from . import model
class MyRef(weakref.ref):

    def __eq__(self, other):
        myref = self()
        return self is other or (myref is not None and myref is other())

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self())
            return self._hash