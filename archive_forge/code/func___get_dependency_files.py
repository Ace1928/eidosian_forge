import sys
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import PySide6
from pathlib import Path
from typing import List
from pkginfo import Wheel
from .. import MAJOR_VERSION, BaseConfig, Config, run_command
from . import (create_recipe, find_lib_dependencies, find_qtlibs_in_wheel,
def __get_dependency_files(self, modules: List[str], arch: str) -> List[zipfile.Path]:
    """
        Based on pysidedeploy_config.modules, returns the
        Qt6{module}_{arch}-android-dependencies.xml file, which contains the various
        dependencies of the module, like permissions, plugins etc
        """
    dependency_files = []
    needed_dependency_files = [f'Qt{MAJOR_VERSION}{module}_{arch}-android-dependencies.xml' for module in modules]
    for dependency_file_name in needed_dependency_files:
        dependency_file = self.qt_libs_path / dependency_file_name
        if dependency_file.exists():
            dependency_files.append(dependency_file)
    logging.info(f'[DEPLOY] The following dependency files were found: {(*dependency_files,)}')
    return dependency_files