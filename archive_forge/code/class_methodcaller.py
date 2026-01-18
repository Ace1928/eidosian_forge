from builtins import abs as _abs
class methodcaller:
    """
    Return a callable object that calls the given method on its operand.
    After f = methodcaller('name'), the call f(r) returns r.name().
    After g = methodcaller('name', 'date', foo=1), the call g(r) returns
    r.name('date', foo=1).
    """
    __slots__ = ('_name', '_args', '_kwargs')

    def __init__(self, name, /, *args, **kwargs):
        self._name = name
        if not isinstance(self._name, str):
            raise TypeError('method name must be a string')
        self._args = args
        self._kwargs = kwargs

    def __call__(self, obj):
        return getattr(obj, self._name)(*self._args, **self._kwargs)

    def __repr__(self):
        args = [repr(self._name)]
        args.extend(map(repr, self._args))
        args.extend(('%s=%r' % (k, v) for k, v in self._kwargs.items()))
        return '%s.%s(%s)' % (self.__class__.__module__, self.__class__.__name__, ', '.join(args))

    def __reduce__(self):
        if not self._kwargs:
            return (self.__class__, (self._name,) + self._args)
        else:
            from functools import partial
            return (partial(self.__class__, self._name, **self._kwargs), self._args)