import logging
from pyside2uic.Compiler.indenter import write_code
from pyside2uic.Compiler.qtproxies import (QtWidgets, QtGui, Literal,
class _CustomWidgetLoader(object):

    def __init__(self):
        self._widgets = {}
        self._usedWidgets = set()

    def addCustomWidget(self, widgetClass, baseClass, module):
        assert widgetClass not in self._widgets
        self._widgets[widgetClass] = (baseClass, module)

    def _resolveBaseclass(self, baseClass):
        try:
            for x in range(0, 10):
                try:
                    return strict_getattr(QtWidgets, baseClass)
                except AttributeError:
                    pass
                baseClass = self._widgets[baseClass][0]
            else:
                raise ValueError('baseclass resolve took too long, check custom widgets')
        except KeyError:
            raise ValueError('unknown baseclass %s' % baseClass)

    def search(self, cls):
        try:
            self._usedWidgets.add(cls)
            baseClass = self._resolveBaseclass(self._widgets[cls][0])
            DEBUG('resolved baseclass of %s: %s' % (cls, baseClass))
            return type(cls, (baseClass,), {'module': ''})
        except KeyError:
            return None

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