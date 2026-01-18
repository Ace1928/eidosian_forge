from __future__ import annotations
import enum
import sys
import types
import typing
import warnings
class _ModuleWithDeprecations(types.ModuleType):

    def __init__(self, module: types.ModuleType):
        super().__init__(module.__name__)
        self.__dict__['_module'] = module

    def __getattr__(self, attr: str) -> object:
        obj = getattr(self._module, attr)
        if isinstance(obj, _DeprecatedValue):
            warnings.warn(obj.message, obj.warning_class, stacklevel=2)
            obj = obj.value
        return obj

    def __setattr__(self, attr: str, value: object) -> None:
        setattr(self._module, attr, value)

    def __delattr__(self, attr: str) -> None:
        obj = getattr(self._module, attr)
        if isinstance(obj, _DeprecatedValue):
            warnings.warn(obj.message, obj.warning_class, stacklevel=2)
        delattr(self._module, attr)

    def __dir__(self) -> typing.Sequence[str]:
        return ['_module'] + dir(self._module)