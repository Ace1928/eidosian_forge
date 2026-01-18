import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
class ModuleLoader(BaseLoader):
    """This loader loads templates from precompiled templates.

    Example usage:

    >>> loader = ChoiceLoader([
    ...     ModuleLoader('/path/to/compiled/templates'),
    ...     FileSystemLoader('/path/to/templates')
    ... ])

    Templates can be precompiled with :meth:`Environment.compile_templates`.
    """
    has_source_access = False

    def __init__(self, path: t.Union[str, os.PathLike, t.Sequence[t.Union[str, os.PathLike]]]) -> None:
        package_name = f'_jinja2_module_templates_{id(self):x}'
        mod = _TemplateModule(package_name)
        if not isinstance(path, abc.Iterable) or isinstance(path, str):
            path = [path]
        mod.__path__ = [os.fspath(p) for p in path]
        sys.modules[package_name] = weakref.proxy(mod, lambda x: sys.modules.pop(package_name, None))
        self.module = mod
        self.package_name = package_name

    @staticmethod
    def get_template_key(name: str) -> str:
        return 'tmpl_' + sha1(name.encode('utf-8')).hexdigest()

    @staticmethod
    def get_module_filename(name: str) -> str:
        return ModuleLoader.get_template_key(name) + '.py'

    @internalcode
    def load(self, environment: 'Environment', name: str, globals: t.Optional[t.MutableMapping[str, t.Any]]=None) -> 'Template':
        key = self.get_template_key(name)
        module = f'{self.package_name}.{key}'
        mod = getattr(self.module, module, None)
        if mod is None:
            try:
                mod = __import__(module, None, None, ['root'])
            except ImportError as e:
                raise TemplateNotFound(name) from e
            sys.modules.pop(module, None)
        if globals is None:
            globals = {}
        return environment.template_class.from_module_dict(environment, mod.__dict__, globals)