from __future__ import annotations
import os
import typing as t
import astroid
from pylint.checkers import BaseChecker
def _check_importfrom(self, node, modname, names):
    """Check the imports on the specified import from node."""
    self._check_module_import(node, modname)
    entry = self.unwanted_imports.get(modname)
    if not entry:
        return
    for name in names:
        if entry.applies_to(self.linter.current_file, name[0]):
            self.add_message(self.BAD_IMPORT_FROM, args=(name[0], entry.alternative, modname), node=node)