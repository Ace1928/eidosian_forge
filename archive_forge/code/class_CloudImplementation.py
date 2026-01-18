import abc
from collections import defaultdict
import collections.abc
from contextlib import contextmanager
import os
from pathlib import (  # type: ignore
import shutil
import sys
from typing import (
from urllib.parse import urlparse
from warnings import warn
from cloudpathlib.enums import FileCacheMode
from . import anypath
from .exceptions import (
class CloudImplementation:
    name: str
    dependencies_loaded: bool = True
    _client_class: Type['Client']
    _path_class: Type['CloudPath']

    def validate_completeness(self) -> None:
        expected = ['client_class', 'path_class']
        missing = [cls for cls in expected if getattr(self, f'_{cls}') is None]
        if missing:
            raise IncompleteImplementationError(f'Implementation is missing registered components: {missing}')
        if not self.dependencies_loaded:
            raise MissingDependenciesError(f"Missing dependencies for {self._client_class.__name__}. You can install them with 'pip install cloudpathlib[{self.name}]'.")

    @property
    def client_class(self) -> Type['Client']:
        self.validate_completeness()
        return self._client_class

    @property
    def path_class(self) -> Type['CloudPath']:
        self.validate_completeness()
        return self._path_class