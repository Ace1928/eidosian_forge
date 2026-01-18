import logging
from typing import List
from pathlib import Path
from . import extract_and_copy_jar, get_wheel_android_arch
from .. import Config, find_pyside_modules
def _find_and_set_pysidemodules(self):
    self.modules = find_pyside_modules(project_dir=self.project_dir, extra_ignore_dirs=self.extra_ignore_dirs, project_data=self.project_data)
    logging.info(f'The following PySide modules were found from the python files of the project {self.modules}')