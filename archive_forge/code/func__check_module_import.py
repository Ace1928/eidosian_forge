from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def _check_module_import(self, node, modname):
    """Check the module import on the given import or import from node."""
    if not is_module_path(self.linter.current_file):
        return
    if modname == 'ansible.module_utils' or modname.startswith('ansible.module_utils.'):
        return
    if modname == 'ansible' or modname.startswith('ansible.'):
        self.add_message(self.BAD_MODULE_IMPORT, args=(modname,), node=node)