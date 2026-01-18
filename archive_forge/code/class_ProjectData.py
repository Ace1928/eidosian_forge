import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
class ProjectData:

    def __init__(self, project_file: Path) -> None:
        """Parse the project."""
        self._project_file = project_file
        self._sub_projects_files: List[Path] = []
        self._files: List[Path] = []
        self._qml_files: List[Path] = []
        self.main_file: Path = None
        self._python_files: List[Path] = []
        self._ui_files: List[Path] = []
        self._qrc_files: List[Path] = []
        with project_file.open('r') as pyf:
            pyproject = json.load(pyf)
            for f in pyproject['files']:
                file = Path(project_file.parent / f)
                if file.suffix == PROJECT_FILE_SUFFIX:
                    self._sub_projects_files.append(file)
                else:
                    self._files.append(file)
                    if file.suffix == '.qml':
                        self._qml_files.append(file)
                    elif is_python_file(file):
                        if file.stem == 'main':
                            self.main_file = file
                        self._python_files.append(file)
                    elif file.suffix == '.ui':
                        self._ui_files.append(file)
                    elif file.suffix == '.qrc':
                        self._qrc_files.append(file)
        if not self.main_file:
            self._find_main_file()

    @property
    def project_file(self):
        return self._project_file

    @property
    def files(self):
        return self._files

    @property
    def main_file(self):
        return self._main_file

    @main_file.setter
    def main_file(self, main_file):
        self._main_file = main_file

    @property
    def python_files(self):
        return self._python_files

    @property
    def ui_files(self):
        return self._ui_files

    @property
    def qrc_files(self):
        return self._qrc_files

    @property
    def qml_files(self):
        return self._qml_files

    @property
    def sub_projects_files(self):
        return self._sub_projects_files

    def _find_main_file(self) -> str:
        """Find the entry point file containing the main function"""

        def is_main(file):
            return '__main__' in file.read_text(encoding='utf-8')
        if not self.main_file:
            for python_file in self.python_files:
                if is_main(python_file):
                    self.main_file = python_file
                    return str(python_file)
        print(f'Python file with main function not found. Add the file to {self.project_file}', file=sys.stderr)
        sys.exit(1)