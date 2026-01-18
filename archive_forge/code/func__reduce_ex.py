def _reduce_ex(self, proto):
    assert proto < 2
    cls = self.__class__
    for base in cls.__mro__:
        if hasattr(base, '__flags__') and (not base.__flags__ & _HEAPTYPE):
            break
        new = base.__new__
        if isinstance(new, _new_type) and new.__self__ is base:
            break
    else:
        base = object
    if base is object:
        state = None
    else:
        if base is cls:
            raise TypeError(f'cannot pickle {cls.__name__!r} object')
        state = base(self)
    args = (cls, base, state)
    try:
        getstate = self.__getstate__
    except AttributeError:
        if getattr(self, '__slots__', None):
            raise TypeError(f'cannot pickle {cls.__name__!r} object: a class that defines __slots__ without defining __getstate__ cannot be pickled with protocol {proto}') from None
        try:
            dict = self.__dict__
        except AttributeError:
            dict = None
    else:
        if type(self).__getstate__ is object.__getstate__ and getattr(self, '__slots__', None):
            raise TypeError('a class that defines __slots__ without defining __getstate__ cannot be pickled')
        dict = getstate()
    if dict:
        return (_reconstructor, args, dict)
    else:
        return (_reconstructor, args)