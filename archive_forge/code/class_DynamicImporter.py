import sys
import warnings
import importlib
from contextlib import contextmanager
import gi
from ._gi import Repository, RepositoryError
from ._gi import PyGIWarning
from .module import get_introspection_module
from .overrides import load_overrides
class DynamicImporter(object):

    def __init__(self, path):
        self.path = path

    def _find_module_check(self, fullname):
        if not fullname.startswith(self.path):
            return False
        path, namespace = fullname.rsplit('.', 1)
        return path == self.path

    def find_spec(self, fullname, path=None, target=None):
        if self._find_module_check(fullname):
            return importlib.util.spec_from_loader(fullname, self)

    def find_module(self, fullname, path=None):
        if self._find_module_check(fullname):
            return self

    def create_module(self, spec):
        path, namespace = spec.name.rsplit('.', 1)
        if not repository.is_registered(namespace) and (not repository.enumerate_versions(namespace)):
            raise ImportError('cannot import name %s, introspection typelib not found' % namespace)
        stacklevel = get_import_stacklevel(import_hook=True)
        with _check_require_version(namespace, stacklevel=stacklevel):
            try:
                introspection_module = get_introspection_module(namespace)
            except RepositoryError as e:
                raise ImportError(e)
            for dep in repository.get_immediate_dependencies(namespace):
                importlib.import_module('gi.repository.' + dep.split('-')[0])
            dynamic_module = load_overrides(introspection_module)
        return dynamic_module

    def exec_module(self, fullname):
        pass