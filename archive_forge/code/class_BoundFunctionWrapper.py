import sys
import operator
import inspect
class BoundFunctionWrapper(_FunctionWrapperBase):

    def __call__(*args, **kwargs):

        def _unpack_self(self, *args):
            return (self, args)
        self, args = _unpack_self(*args)
        if self._self_enabled is not None:
            if callable(self._self_enabled):
                if not self._self_enabled():
                    return self.__wrapped__(*args, **kwargs)
            elif not self._self_enabled:
                return self.__wrapped__(*args, **kwargs)
        if self._self_binding == 'function':
            if self._self_instance is None:
                if not args:
                    raise TypeError('missing 1 required positional argument')
                instance, args = (args[0], args[1:])
                wrapped = PartialCallableObjectProxy(self.__wrapped__, instance)
                return self._self_wrapper(wrapped, instance, args, kwargs)
            return self._self_wrapper(self.__wrapped__, self._self_instance, args, kwargs)
        else:
            instance = getattr(self.__wrapped__, '__self__', None)
            return self._self_wrapper(self.__wrapped__, instance, args, kwargs)