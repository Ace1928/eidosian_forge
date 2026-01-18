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
def __find_dependent_qt_modules(self, pysidedeploy_config: Config):
    """
        Given pysidedeploy_config.modules, find all the other dependent Qt modules. This is
        done by using llvm-readobj (readelf) to find the dependent libraries from the module
        library.
        """
    dependent_modules = set()
    all_dependencies = set()
    lib_pattern = re.compile(f'libQt6(?P<mod_name>.*)_{self.arch}')
    llvm_readobj = get_llvm_readobj(pysidedeploy_config.ndk_path)
    if not llvm_readobj.exists():
        raise FileNotFoundError(f'[DEPLOY] {llvm_readobj} does not exist.Finding Qt dependencies failed')
    archive = zipfile.ZipFile(pysidedeploy_config.wheel_pyside)
    lib_path_suffix = Path(str(self.qt_libs_path)).relative_to(pysidedeploy_config.wheel_pyside)
    with tempfile.TemporaryDirectory() as tmpdir:
        archive.extractall(tmpdir)
        qt_libs_tmpdir = Path(tmpdir) / lib_path_suffix
        for module_name in pysidedeploy_config.modules:
            qt_module_path = qt_libs_tmpdir / f'libQt6{module_name}_{self.arch}.so'
            if not qt_module_path.exists():
                raise FileNotFoundError(f'[DEPLOY] libQt6{module_name}_{self.arch}.so not found inside the wheel')
            find_lib_dependencies(llvm_readobj=llvm_readobj, lib_path=qt_module_path, dry_run=pysidedeploy_config.dry_run, used_dependencies=all_dependencies)
    for dependency in all_dependencies:
        match = lib_pattern.search(dependency)
        if match:
            module = match.group('mod_name')
            if module not in pysidedeploy_config.modules:
                dependent_modules.add(module)
    dependent_modules = [module for module in dependent_modules if module in ALL_PYSIDE_MODULES]
    dependent_modules_str = ','.join(dependent_modules)
    logging.info(f'[DEPLOY] The following extra dependencies were found: {dependent_modules_str}')
    return list(dependent_modules)