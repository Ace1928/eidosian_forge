import importlib.abc
import importlib.util
import sys
import types
from importlib import import_module
from .importstring import import_item
class ShimModule(types.ModuleType):

    def __init__(self, *args, **kwargs):
        self._mirror = kwargs.pop('mirror')
        src = kwargs.pop('src', None)
        if src:
            kwargs['name'] = src.rsplit('.', 1)[-1]
        super(ShimModule, self).__init__(*args, **kwargs)
        if src:
            sys.meta_path.append(ShimImporter(src=src, mirror=self._mirror))

    @property
    def __path__(self):
        return []

    @property
    def __spec__(self):
        """Don't produce __spec__ until requested"""
        return import_module(self._mirror).__spec__

    def __dir__(self):
        return dir(import_module(self._mirror))

    @property
    def __all__(self):
        """Ensure __all__ is always defined"""
        mod = import_module(self._mirror)
        try:
            return mod.__all__
        except AttributeError:
            return [name for name in dir(mod) if not name.startswith('_')]

    def __getattr__(self, key):
        name = '%s.%s' % (self._mirror, key)
        try:
            return import_item(name)
        except ImportError as e:
            raise AttributeError(key) from e

    def __repr__(self):
        try:
            return self.__getattr__('__repr__')()
        except AttributeError:
            return f'<ShimModule for {self._mirror!r}>'