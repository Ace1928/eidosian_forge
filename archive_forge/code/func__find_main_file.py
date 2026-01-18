import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
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