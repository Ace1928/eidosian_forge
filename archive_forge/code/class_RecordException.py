import pickle
from collections import namedtuple
class RecordException(namedtuple('RecordException', ('type', 'value', 'traceback'))):

    def __repr__(self):
        return '(type=%r, value=%r, traceback=%r)' % (self.type, self.value, self.traceback)

    def __reduce__(self):
        try:
            pickled_value = pickle.dumps(self.value)
        except Exception:
            return (RecordException, (self.type, None, None))
        else:
            return (RecordException._from_pickled_value, (self.type, pickled_value, None))

    @classmethod
    def _from_pickled_value(cls, type_, pickled_value, traceback_):
        try:
            value = pickle.loads(pickled_value)
        except Exception:
            return cls(type_, None, traceback_)
        else:
            return cls(type_, value, traceback_)