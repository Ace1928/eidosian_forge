from __future__ import annotations
import ast
import os
import re
import typing as t
from ..io import (
from ..util import (
from ..data import (
from ..target import (
class ModuleUtilFinder(ast.NodeVisitor):
    """AST visitor to find valid module_utils imports."""

    def __init__(self, path: str, module_utils: set[str]) -> None:
        self.path = path
        self.module_utils = module_utils
        self.imports: set[str] = set()
        if path.endswith('/__init__.py'):
            path = os.path.split(path)[0]
        if path.startswith('lib/ansible/module_utils/'):
            package = os.path.split(path)[0].replace('/', '.')[4:]
            if package != 'ansible.module_utils' and package not in VIRTUAL_PACKAGES:
                self.add_import(package, 0)
        self.module = None
        if data_context().content.is_ansible:
            path_map = (('^lib/ansible/', 'ansible/'), ('^test/lib/ansible_test/_util/controller/sanity/validate-modules/', 'validate_modules/'), ('^test/units/', 'test/units/'), ('^test/lib/ansible_test/_internal/', 'ansible_test/_internal/'), ('^test/integration/targets/.*/ansible_collections/(?P<ns>[^/]*)/(?P<col>[^/]*)/', 'ansible_collections/\\g<ns>/\\g<col>/'), ('^test/integration/targets/.*/library/', 'ansible/modules/'))
            for pattern, replacement in path_map:
                if re.search(pattern, self.path):
                    revised_path = re.sub(pattern, replacement, self.path)
                    self.module = path_to_module(revised_path)
                    break
        else:
            self.module = path_to_module(os.path.join(data_context().content.collection.directory, self.path))

    def visit_Import(self, node: ast.Import) -> None:
        """Visit an import node."""
        self.generic_visit(node)
        self.add_imports([alias.name for alias in node.names], node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit an import from node."""
        self.generic_visit(node)
        if not node.module:
            return
        module = relative_to_absolute(node.module, node.level, self.module, self.path, node.lineno)
        if not module.startswith('ansible'):
            return
        self.add_imports(['%s.%s' % (module, alias.name) for alias in node.names], node.lineno)

    def add_import(self, name: str, line_number: int) -> None:
        """Record the specified import."""
        import_name = name
        while self.is_module_util_name(name):
            if name in self.module_utils:
                if name not in self.imports:
                    display.info('%s:%d imports module_utils: %s' % (self.path, line_number, name), verbosity=5)
                    self.imports.add(name)
                return
            name = '.'.join(name.split('.')[:-1])
        if is_subdir(self.path, data_context().content.test_path):
            return
        display.warning('%s:%d Invalid module_utils import: %s' % (self.path, line_number, import_name))

    def add_imports(self, names: list[str], line_no: int) -> None:
        """Add the given import names if they are module_utils imports."""
        for name in names:
            if self.is_module_util_name(name):
                self.add_import(name, line_no)

    @staticmethod
    def is_module_util_name(name: str) -> bool:
        """Return True if the given name is a module_util name for the content under test. External module_utils are ignored."""
        if data_context().content.is_ansible and name.startswith('ansible.module_utils.'):
            return True
        if data_context().content.collection and name.startswith('ansible_collections.%s.plugins.module_utils.' % data_context().content.collection.full_name):
            return True
        return False