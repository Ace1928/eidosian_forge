import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def _qml_module_check(self):
    """Run a pre-check on Python source files and find the ones with QML
        decorators (representing a QML module)."""
    if not opt_qml_module and (not self.project.qml_files):
        return
    for file in self.project.files:
        if is_python_file(file):
            has_class, data = check_qml_decorators(file)
            if has_class:
                self._qml_module_sources.append(file)
                if data:
                    self._qml_project_data = data
    if not self._qml_module_sources:
        return
    if not self._qml_project_data:
        print('Detected QML-decorated files, but was unable to detect QML_IMPORT_NAME')
        sys.exit(1)
    self._qml_module_dir = self.project.project_file.parent
    for uri_dir in self._qml_project_data.import_name.split('.'):
        self._qml_module_dir /= uri_dir
    print(self._qml_module_dir)
    self._qml_dir_file = self._qml_module_dir / QMLDIR_FILE
    if not opt_quiet:
        count = len(self._qml_module_sources)
        print(f'{self.project.project_file.name}, {count} QML file(s), {self._qml_project_data}')