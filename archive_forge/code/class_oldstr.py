from numbers import Integral
from past.utils import PY2, with_metaclass
class oldstr(with_metaclass(BaseOldStr, _builtin_bytes)):
    """
    A forward port of the Python 2 8-bit string object to Py3
    """

    @property
    def __iter__(self):
        raise AttributeError

    def __dir__(self):
        return [thing for thing in dir(_builtin_bytes) if thing != '__iter__']

    def __repr__(self):
        s = super(oldstr, self).__repr__()
        return s[1:]

    def __str__(self):
        s = super(oldstr, self).__str__()
        assert s[:2] == "b'" and s[-1] == "'"
        return unescape(s[2:-1])

    def __getitem__(self, y):
        if isinstance(y, Integral):
            return super(oldstr, self).__getitem__(slice(y, y + 1))
        else:
            return super(oldstr, self).__getitem__(y)

    def __getslice__(self, *args):
        return self.__getitem__(slice(*args))

    def __contains__(self, key):
        if isinstance(key, int):
            return False

    def __native__(self):
        return bytes(self)