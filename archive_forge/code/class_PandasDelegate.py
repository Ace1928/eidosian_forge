from __future__ import annotations
from typing import (
import warnings
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level
class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    def _delegate_property_get(self, name: str, *args, **kwargs):
        raise TypeError(f'You cannot access the property {name}')

    def _delegate_property_set(self, name: str, value, *args, **kwargs):
        raise TypeError(f'The property {name} cannot be set')

    def _delegate_method(self, name: str, *args, **kwargs):
        raise TypeError(f'You cannot call method {name}')

    @classmethod
    def _add_delegate_accessors(cls, delegate, accessors: list[str], typ: str, overwrite: bool=False, accessor_mapping: Callable[[str], str]=lambda x: x, raise_on_missing: bool=True) -> None:
        """
        Add accessors to cls from the delegate class.

        Parameters
        ----------
        cls
            Class to add the methods/properties to.
        delegate
            Class to get methods/properties and doc-strings.
        accessors : list of str
            List of accessors to add.
        typ : {'property', 'method'}
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable, default lambda x: x
            Callable to map the delegate's function to the cls' function.
        raise_on_missing: bool, default True
            Raise if an accessor does not exist on delegate.
            False skips the missing accessor.
        """

        def _create_delegator_property(name: str):

            def _getter(self):
                return self._delegate_property_get(name)

            def _setter(self, new_values):
                return self._delegate_property_set(name, new_values)
            _getter.__name__ = name
            _setter.__name__ = name
            return property(fget=_getter, fset=_setter, doc=getattr(delegate, accessor_mapping(name)).__doc__)

        def _create_delegator_method(name: str):

            def f(self, *args, **kwargs):
                return self._delegate_method(name, *args, **kwargs)
            f.__name__ = name
            f.__doc__ = getattr(delegate, accessor_mapping(name)).__doc__
            return f
        for name in accessors:
            if not raise_on_missing and getattr(delegate, accessor_mapping(name), None) is None:
                continue
            if typ == 'property':
                f = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)