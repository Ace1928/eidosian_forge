def __coerce__(self, other):
    if isinstance(other, IntSet):
        return (self, other)
    elif isinstance(other, (int, tuple)):
        try:
            return (self, self.__class__(other))
        except TypeError:
            return NotImplemented
    elif isinstance(other, list):
        try:
            return (self, self.__class__(*other))
        except TypeError:
            return NotImplemented
    return NotImplemented