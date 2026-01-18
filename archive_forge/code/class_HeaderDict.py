import warnings
class HeaderDict(dict):
    """
    This represents response headers.  It handles the headers as a
    dictionary, with case-insensitive keys.

    Also there is an ``.add(key, value)`` method, which sets the key,
    or adds the value to the current value (turning it into a list if
    necessary).

    For passing to WSGI there is a ``.headeritems()`` method which is
    like ``.items()`` but unpacks value that are lists.  It also
    handles encoding -- all headers are encoded in ASCII (if they are
    unicode).

    @@: Should that encoding be ISO-8859-1 or UTF-8?  I'm not sure
    what the spec says.
    """

    def __getitem__(self, key):
        return dict.__getitem__(self, self.normalize(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, self.normalize(key), value)

    def __delitem__(self, key):
        dict.__delitem__(self, self.normalize(key))

    def __contains__(self, key):
        return dict.__contains__(self, self.normalize(key))
    has_key = __contains__

    def get(self, key, failobj=None):
        return dict.get(self, self.normalize(key), failobj)

    def setdefault(self, key, failobj=None):
        return dict.setdefault(self, self.normalize(key), failobj)

    def pop(self, key, *args):
        return dict.pop(self, self.normalize(key), *args)

    def update(self, other):
        for key in other:
            self[self.normalize(key)] = other[key]

    def normalize(self, key):
        return str(key).lower().strip()

    def add(self, key, value):
        key = self.normalize(key)
        if key in self:
            if isinstance(self[key], list):
                self[key].append(value)
            else:
                self[key] = [self[key], value]
        else:
            self[key] = value

    def headeritems(self):
        result = []
        for key, value in self.items():
            if isinstance(value, list):
                for v in value:
                    result.append((key, str(v)))
            else:
                result.append((key, str(value)))
        return result

    def fromlist(cls, seq):
        self = cls()
        for name, value in seq:
            self.add(name, value)
        return self
    fromlist = classmethod(fromlist)