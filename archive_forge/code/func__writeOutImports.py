import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
def _writeOutImports(self):
    for module in self._modules:
        module._writeImportCode()