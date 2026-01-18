import sys
import operator
import inspect
class PartialCallableObjectProxy(ObjectProxy):

    def __init__(*args, **kwargs):

        def _unpack_self(self, *args):
            return (self, args)
        self, args = _unpack_self(*args)
        if len(args) < 1:
            raise TypeError('partial type takes at least one argument')
        wrapped, args = (args[0], args[1:])
        if not callable(wrapped):
            raise TypeError('the first argument must be callable')
        super(PartialCallableObjectProxy, self).__init__(wrapped)
        self._self_args = args
        self._self_kwargs = kwargs

    def __call__(*args, **kwargs):

        def _unpack_self(self, *args):
            return (self, args)
        self, args = _unpack_self(*args)
        _args = self._self_args + args
        _kwargs = dict(self._self_kwargs)
        _kwargs.update(kwargs)
        return self.__wrapped__(*_args, **_kwargs)