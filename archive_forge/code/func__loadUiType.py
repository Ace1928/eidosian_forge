import os
import re
import subprocess
import sys
import time
import warnings
from . import QtCore, QtGui, QtWidgets, compat
from . import internals
def _loadUiType(uiFile):
    """
    PySide lacks a "loadUiType" command like PyQt4's, so we have to convert
    the ui file to py code in-memory first and then execute it in a
    special frame to retrieve the form_class.

    from stackoverflow: http://stackoverflow.com/a/14195313/3781327

    seems like this might also be a legitimate solution, but I'm not sure
    how to make PyQt4 and pyside look the same...
        http://stackoverflow.com/a/8717832
    """
    pyside2uic = None
    if QT_LIB == PYSIDE2:
        try:
            import pyside2uic
        except ImportError:
            pyside2uic = None
        if pyside2uic is None:
            pyside2version = tuple(map(int, PySide2.__version__.split('.')))
            if (5, 14) <= pyside2version < (5, 14, 2, 2):
                warnings.warn('For UI compilation, it is recommended to upgrade to PySide >= 5.15', RuntimeWarning, stacklevel=2)
    import xml.etree.ElementTree as xml
    parsed = xml.parse(uiFile)
    widget_class = parsed.find('widget').get('class')
    form_class = parsed.find('class').text
    if pyside2uic is None:
        uic_executable = QT_LIB.lower() + '-uic'
        uipy = subprocess.check_output([uic_executable, uiFile])
    else:
        o = _StringIO()
        with open(uiFile, 'r') as f:
            pyside2uic.compileUi(f, o, indent=0)
        uipy = o.getvalue()
    pyc = compile(uipy, '<string>', 'exec')
    frame = {}
    exec(pyc, frame)
    form_class = frame['Ui_%s' % form_class]
    base_class = eval('QtWidgets.%s' % widget_class)
    return (form_class, base_class)