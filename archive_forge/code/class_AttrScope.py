import threading
import warnings
from collections import defaultdict
from .base import string_types, classproperty, with_metaclass, _MXClassPropertyMetaClass
class AttrScope(with_metaclass(_MXClassPropertyMetaClass, object)):
    """Attribute manager for scoping.

    User can also inherit this object to change naming behavior.

    Parameters
    ----------
    kwargs
        The attributes to set for all symbol creations in the scope.
    """
    _current = threading.local()
    _subgraph_names = defaultdict(int)

    def __init__(self, **kwargs):
        self._old_scope = None
        for value in kwargs.values():
            if not isinstance(value, string_types):
                raise ValueError('Attributes need to be string')
        self._attr = kwargs

    def get(self, attr):
        """
        Get the attribute dict given the attribute set by the symbol.

        Parameters
        ----------
        attr : dict of string to string
            The attribute passed in by user during symbol creation.

        Returns
        -------
        attr : dict of string to string
            Updated attributes to add other scope related attributes.
        """
        if self._attr:
            ret = self._attr.copy()
            if attr:
                ret.update(attr)
            return ret
        else:
            return attr if attr else {}

    def __enter__(self):
        if not hasattr(AttrScope._current, 'value'):
            AttrScope._current.value = AttrScope()
        self._old_scope = AttrScope._current.value
        attr = AttrScope._current.value._attr.copy()
        attr.update(self._attr)
        self._attr = attr
        AttrScope._current.value = self
        return self

    def __exit__(self, ptype, value, trace):
        assert self._old_scope
        AttrScope._current.value = self._old_scope

    @classproperty
    def current(cls):
        warnings.warn('AttrScope.current has been deprecated. It is advised to use the `with` statement with AttrScope.', DeprecationWarning)
        if not hasattr(AttrScope._current, 'value'):
            cls._current.value = AttrScope()
        return cls._current.value

    @current.setter
    def current(cls, val):
        warnings.warn('AttrScope.current has been deprecated. It is advised to use the `with` statement with AttrScope.', DeprecationWarning)
        cls._current.value = val