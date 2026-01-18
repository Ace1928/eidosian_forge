import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def _regenerate_qmldir(self):
    """Regenerate the 'qmldir' file."""
    if opt_dry_run or not self._qml_dir_file:
        return
    if opt_force or requires_rebuild(self._qml_module_sources, self._qml_dir_file):
        with self._qml_dir_file.open('w') as qf:
            qf.write(f'module {self._qml_project_data.import_name}\n')
            for f in self._qml_module_dir.glob('*.qmltypes'):
                qf.write(f'typeinfo {f.name}\n')