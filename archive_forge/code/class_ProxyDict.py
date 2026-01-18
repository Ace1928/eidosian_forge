from typing import Dict, TypeVar, Optional, Union, Any, TYPE_CHECKING
from .registry import (
from lazyops.types.lazydict import LazyDict, RT
class ProxyDict(LazyDict):
    """
    A Proxy Dictionary that lazily defers initialization of a client until it is called

    These are non-global clients
    """
    initialize_objects: Optional[bool] = True
    exclude_schema_attrs: Optional[bool] = True
    proxy_schema: Optional[Dict[str, str]] = None

    def __init__(self, **kwargs):
        if self.proxy_schema is None:
            raise NotImplementedError('Proxy Schema not set')
        self._dict = self.proxy_schema
        if self.exclude_schema_attrs:
            self.excluded_attrs = list(self.proxy_schema.keys())
        self.post_init(**kwargs)

    def post_init(self, **kwargs):
        """
        Post Initialization to be overwritten by the subclass
        """
        pass

    def obj_initializer(self, name: str, obj: RT, **kwargs) -> RT:
        """
        Returns the object initializer

        - Can be overwritten by the subclass to modify the object initialization
        """
        return obj(**kwargs)

    def get_or_init(self, name: str, default: Optional[RT]=None) -> RT:
        """
        Get an attribute from the dictionary
        If it does not exist, set it to the default value
        """
        if name not in self._dict:
            if default:
                self._dict[name] = default
            else:
                raise ValueError(f'Default value for {name} is None')
        from lazyops.utils.lazy import lazy_import
        if isinstance(self._dict[name], str):
            self._dict[name] = lazy_import(self._dict[name])
            if self.initialize_objects:
                self._dict[name] = self.obj_initializer(name, self._dict[name])
        elif isinstance(self._dict[name], tuple):
            obj_class, kwargs = self._dict[name]
            if isinstance(obj_class, str):
                obj_class = lazy_import(obj_class)
            for k, v in kwargs.items():
                if callable(v):
                    kwargs[k] = v()
            self._dict[name] = self.obj_initializer(name, obj_class, **kwargs)
        elif isinstance(self._dict[name], dict):
            self._dict[name] = self.obj_initializer(name, type(self), **self._dict[name])
        return self._dict[name]