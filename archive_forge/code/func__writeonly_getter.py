from . import _gi
from ._constants import \
def _writeonly_getter(self, instance):
    self._exc = TypeError('%s property of %s is write-only' % (self.name, type(instance).__name__))