import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def _writeImportCode(self):
    imports = {}
    for widget in self._usedWidgets:
        _, module = self._widgets[widget]
        imports.setdefault(module, []).append(widget)
    for module, classes in imports.items():
        parts = module.split('.')
        if len(parts) == 2 and (not parts[0].startswith('PySide2')) and (parts[0] in pyside2_modules):
            module = 'PySide2.{}'.format(parts[0])
        write_code('from %s import %s' % (module, ', '.join(classes)))