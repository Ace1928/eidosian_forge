import weakref
from eventlet import greenthread
class _localbase:
    __slots__ = ('_local__args', '_local__greens')

    def __new__(cls, *args, **kw):
        self = object.__new__(cls)
        object.__setattr__(self, '_local__args', (args, kw))
        object.__setattr__(self, '_local__greens', weakref.WeakKeyDictionary())
        if (args or kw) and cls.__init__ is object.__init__:
            raise TypeError('Initialization arguments are not supported')
        return self