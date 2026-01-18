import pickle
from collections import namedtuple
class RecordFile:
    __slots__ = ('name', 'path')

    def __init__(self, name, path):
        self.name = name
        self.path = path

    def __repr__(self):
        return '(name=%r, path=%r)' % (self.name, self.path)

    def __format__(self, spec):
        return self.name.__format__(spec)