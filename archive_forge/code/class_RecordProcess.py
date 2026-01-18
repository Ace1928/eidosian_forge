import pickle
from collections import namedtuple
class RecordProcess:
    __slots__ = ('id', 'name')

    def __init__(self, id_, name):
        self.id = id_
        self.name = name

    def __repr__(self):
        return '(id=%r, name=%r)' % (self.id, self.name)

    def __format__(self, spec):
        return self.id.__format__(spec)