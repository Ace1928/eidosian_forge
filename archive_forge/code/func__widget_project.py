import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import List, Tuple
from PySide6.QtWidgets import QApplication, QMainWindow
import QtQuick.Controls
from pathlib import Path
from PySide6.QtGui import QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtQml import QQmlApplicationEngine
def _widget_project() -> Project:
    """Create a (form-less) widgets project."""
    main_py = _WIDGET_IMPORTS + '\n\n' + _WIDGET_CLASS_DEFINITION + '\n\n' + _WIDGET_MAIN
    return [('main.py', main_py)]