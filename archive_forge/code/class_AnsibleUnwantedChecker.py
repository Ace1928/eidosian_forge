from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
class AnsibleUnwantedChecker(BaseChecker):
    """Checker for unwanted imports and functions."""
    __implements__ = (IAstroidChecker,)
    name = 'unwanted'
    BAD_IMPORT = 'ansible-bad-import'
    BAD_IMPORT_FROM = 'ansible-bad-import-from'
    BAD_FUNCTION = 'ansible-bad-function'
    BAD_MODULE_IMPORT = 'ansible-bad-module-import'
    msgs = dict(E5101=('Import %s instead of %s', BAD_IMPORT, 'Identifies imports which should not be used.'), E5102=('Import %s from %s instead of %s', BAD_IMPORT_FROM, 'Identifies imports which should not be used.'), E5103=('Call %s instead of %s', BAD_FUNCTION, 'Identifies functions which should not be used.'), E5104=('Import external package or ansible.module_utils not %s', BAD_MODULE_IMPORT, 'Identifies imports which should not be used.'))
    unwanted_imports = dict(urllib2=UnwantedEntry('ansible.module_utils.urls', ignore_paths=('/lib/ansible/module_utils/urls.py',)), collections=UnwantedEntry('ansible.module_utils.six.moves.collections_abc', names=('MappingView', 'ItemsView', 'KeysView', 'ValuesView', 'Mapping', 'MutableMapping', 'Sequence', 'MutableSequence', 'Set', 'MutableSet', 'Container', 'Hashable', 'Sized', 'Callable', 'Iterable', 'Iterator')))
    unwanted_functions = {'tempfile.mktemp': UnwantedEntry('tempfile.mkstemp'), 'posix.chmod': UnwantedEntry('verified_chmod', ansible_test_only=True), 'sys.exit': UnwantedEntry('exit_json or fail_json', ignore_paths=('/lib/ansible/module_utils/basic.py', '/lib/ansible/modules/async_wrapper.py'), modules_only=True), 'builtins.print': UnwantedEntry('module.log or module.debug', ignore_paths=('/lib/ansible/module_utils/basic.py',), modules_only=True)}

    def visit_import(self, node):
        """Visit an import node."""
        for name in node.names:
            self._check_import(node, name[0])

    def visit_importfrom(self, node):
        """Visit an import from node."""
        self._check_importfrom(node, node.modname, node.names)

    def visit_attribute(self, node):
        """Visit an attribute node."""
        last_child = node.last_child()
        if not isinstance(last_child, astroid.node_classes.Name):
            return
        module = last_child.name
        entry = self.unwanted_imports.get(module)
        if entry and entry.names:
            if entry.applies_to(self.linter.current_file, node.attrname):
                self.add_message(self.BAD_IMPORT_FROM, args=(node.attrname, entry.alternative, module), node=node)

    def visit_call(self, node):
        """Visit a call node."""
        try:
            for i in node.func.inferred():
                func = None
                if isinstance(i, astroid.scoped_nodes.FunctionDef) and isinstance(i.parent, astroid.scoped_nodes.Module):
                    func = '%s.%s' % (i.parent.name, i.name)
                if not func:
                    continue
                entry = self.unwanted_functions.get(func)
                if entry and entry.applies_to(self.linter.current_file):
                    self.add_message(self.BAD_FUNCTION, args=(entry.alternative, func), node=node)
        except astroid.exceptions.InferenceError:
            pass

    def _check_import(self, node, modname):
        """Check the imports on the specified import node."""
        self._check_module_import(node, modname)
        entry = self.unwanted_imports.get(modname)
        if not entry:
            return
        if entry.applies_to(self.linter.current_file):
            self.add_message(self.BAD_IMPORT, args=(entry.alternative, modname), node=node)

    def _check_importfrom(self, node, modname, names):
        """Check the imports on the specified import from node."""
        self._check_module_import(node, modname)
        entry = self.unwanted_imports.get(modname)
        if not entry:
            return
        for name in names:
            if entry.applies_to(self.linter.current_file, name[0]):
                self.add_message(self.BAD_IMPORT_FROM, args=(name[0], entry.alternative, modname), node=node)

    def _check_module_import(self, node, modname):
        """Check the module import on the given import or import from node."""
        if not is_module_path(self.linter.current_file):
            return
        if modname == 'ansible.module_utils' or modname.startswith('ansible.module_utils.'):
            return
        if modname == 'ansible' or modname.startswith('ansible.'):
            self.add_message(self.BAD_MODULE_IMPORT, args=(modname,), node=node)