import requests
class _ThreadingDescriptor(object):

    def __init__(self, prop, default):
        self.prop = prop
        self.default = default

    def __get__(self, obj, objtype=None):
        return getattr(obj._thread_local, self.prop, self.default)

    def __set__(self, obj, value):
        setattr(obj._thread_local, self.prop, value)