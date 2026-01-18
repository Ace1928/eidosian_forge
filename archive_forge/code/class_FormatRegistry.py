from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
from .pyutils import get_named_object
class FormatRegistry(Registry[str, Union[Format, Callable[[], Format]]]):
    """Registry specialised for handling formats."""

    def __init__(self, other_registry=None):
        super().__init__()
        self._other_registry = other_registry

    def register(self, key, obj, help=None, info=None, override_existing=False):
        Registry.register(self, key, obj, help=help, info=info, override_existing=override_existing)
        if self._other_registry is not None:
            self._other_registry.register(key, obj, help=help, info=info, override_existing=override_existing)

    def register_lazy(self, key, module_name, member_name, help=None, info=None, override_existing=False):
        Registry.register_lazy(self, key, module_name, member_name, help=help, info=info, override_existing=override_existing)
        if self._other_registry is not None:
            self._other_registry.register_lazy(key, module_name, member_name, help=help, info=info, override_existing=override_existing)

    def remove(self, key):
        super().remove(key)
        if self._other_registry is not None:
            self._other_registry.remove(key)

    def get(self, format_string):
        r = Registry.get(self, format_string)
        if callable(r):
            r = r()
        return r