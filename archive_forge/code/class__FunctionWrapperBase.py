import sys
import operator
import inspect
class _FunctionWrapperBase(ObjectProxy):
    __slots__ = ('_self_instance', '_self_wrapper', '_self_enabled', '_self_binding', '_self_parent')

    def __init__(self, wrapped, instance, wrapper, enabled=None, binding='function', parent=None):
        super(_FunctionWrapperBase, self).__init__(wrapped)
        object.__setattr__(self, '_self_instance', instance)
        object.__setattr__(self, '_self_wrapper', wrapper)
        object.__setattr__(self, '_self_enabled', enabled)
        object.__setattr__(self, '_self_binding', binding)
        object.__setattr__(self, '_self_parent', parent)

    def __get__(self, instance, owner):
        if self._self_parent is None:
            if not inspect.isclass(self.__wrapped__):
                descriptor = self.__wrapped__.__get__(instance, owner)
                return self.__bound_function_wrapper__(descriptor, instance, self._self_wrapper, self._self_enabled, self._self_binding, self)
            return self
        if self._self_instance is None and self._self_binding == 'function':
            descriptor = self._self_parent.__wrapped__.__get__(instance, owner)
            return self._self_parent.__bound_function_wrapper__(descriptor, instance, self._self_wrapper, self._self_enabled, self._self_binding, self._self_parent)
        return self

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
        if self._self_binding in ('function', 'classmethod'):
            if self._self_instance is None:
                instance = getattr(self.__wrapped__, '__self__', None)
                if instance is not None:
                    return self._self_wrapper(self.__wrapped__, instance, args, kwargs)
        return self._self_wrapper(self.__wrapped__, self._self_instance, args, kwargs)

    def __set_name__(self, owner, name):
        if hasattr(self.__wrapped__, '__set_name__'):
            self.__wrapped__.__set_name__(owner, name)

    def __instancecheck__(self, instance):
        return isinstance(instance, self.__wrapped__)

    def __subclasscheck__(self, subclass):
        if hasattr(subclass, '__wrapped__'):
            return issubclass(subclass.__wrapped__, self.__wrapped__)
        else:
            return issubclass(subclass, self.__wrapped__)