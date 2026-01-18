import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
def _find_and_set_qtquick_modules(self):
    """Identify if QtQuick is used in QML files and add them as dependency
        """
    extra_modules = []
    if 'QtQuick' in self.qml_modules:
        extra_modules.append('Quick')
    if 'QtQuick.Controls' in self.qml_modules:
        extra_modules.append('QuickControls2')
    self.modules += extra_modules