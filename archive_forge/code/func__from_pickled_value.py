import pickle
from collections import namedtuple
@classmethod
def _from_pickled_value(cls, type_, pickled_value, traceback_):
    try:
        value = pickle.loads(pickled_value)
    except Exception:
        return cls(type_, None, traceback_)
    else:
        return cls(type_, value, traceback_)