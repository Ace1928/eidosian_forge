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
class DictLoader(BaseLoader):
    """Loads a template from a Python dict mapping template names to
    template source.  This loader is useful for unittesting:

    >>> loader = DictLoader({'index.html': 'source here'})

    Because auto reloading is rarely useful this is disabled per default.
    """

    def __init__(self, mapping: t.Mapping[str, str]) -> None:
        self.mapping = mapping

    def get_source(self, environment: 'Environment', template: str) -> t.Tuple[str, None, t.Callable[[], bool]]:
        if template in self.mapping:
            source = self.mapping[template]
            return (source, None, lambda: source == self.mapping.get(template))
        raise TemplateNotFound(template)

    def list_templates(self) -> t.List[str]:
        return sorted(self.mapping)