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
class PrefixLoader(BaseLoader):
    """A loader that is passed a dict of loaders where each loader is bound
    to a prefix.  The prefix is delimited from the template by a slash per
    default, which can be changed by setting the `delimiter` argument to
    something else::

        loader = PrefixLoader({
            'app1':     PackageLoader('mypackage.app1'),
            'app2':     PackageLoader('mypackage.app2')
        })

    By loading ``'app1/index.html'`` the file from the app1 package is loaded,
    by loading ``'app2/index.html'`` the file from the second.
    """

    def __init__(self, mapping: t.Mapping[str, BaseLoader], delimiter: str='/') -> None:
        self.mapping = mapping
        self.delimiter = delimiter

    def get_loader(self, template: str) -> t.Tuple[BaseLoader, str]:
        try:
            prefix, name = template.split(self.delimiter, 1)
            loader = self.mapping[prefix]
        except (ValueError, KeyError) as e:
            raise TemplateNotFound(template) from e
        return (loader, name)

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, t.Optional[str], t.Optional[t.Callable[[], bool]]]:
        loader, name = self.get_loader(template)
        try:
            return loader.get_source(environment, name)
        except TemplateNotFound as e:
            raise TemplateNotFound(template) from e

    @internalcode
    def load(self, environment: 'Environment', name: str, globals: t.Optional[t.MutableMapping[str, t.Any]]=None) -> 'Template':
        loader, local_name = self.get_loader(name)
        try:
            return loader.load(environment, local_name, globals)
        except TemplateNotFound as e:
            raise TemplateNotFound(name) from e

    def list_templates(self) -> t.List[str]:
        result = []
        for prefix, loader in self.mapping.items():
            for template in loader.list_templates():
                result.append(prefix + self.delimiter + template)
        return result