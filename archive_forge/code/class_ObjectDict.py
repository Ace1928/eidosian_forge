import weakref
import importlib_metadata
from wsme.exc import ClientSideError
class ObjectDict(object):

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, name):
        return getattr(self.obj, name)