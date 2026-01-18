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
def _ui_form_project() -> Project:
    """Create a Qt Designer .ui form based widgets project."""
    main_py = _WIDGET_IMPORTS + '\nfrom ui_mainwindow import Ui_MainWindow\n\n\n' + _WIDGET_CLASS_DEFINITION + _WIDGET_SETUP_UI_CODE + '\n\n' + _WIDGET_MAIN
    return [('main.py', main_py), ('mainwindow.ui', _MAINWINDOW_FORM)]