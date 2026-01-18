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
class Buildozer:
    dry_run = False

    @staticmethod
    def initialize(pysidedeploy_config: Config):
        project_dir = Path(pysidedeploy_config.project_dir)
        buildozer_spec = project_dir / 'buildozer.spec'
        if buildozer_spec.exists():
            logging.warning(f'[DEPLOY] buildozer.spec already present in {str(project_dir)}.Using it')
            return
        command = [sys.executable, '-m', 'buildozer', 'init']
        run_command(command=command, dry_run=Buildozer.dry_run)
        if not Buildozer.dry_run:
            if not buildozer_spec.exists():
                raise RuntimeError(f'buildozer.spec not found in {Path.cwd()}')
            BuildozerConfig(buildozer_spec, pysidedeploy_config)

    @staticmethod
    def create_executable(mode: str):
        command = [sys.executable, '-m', 'buildozer', 'android', mode]
        run_command(command=command, dry_run=Buildozer.dry_run)