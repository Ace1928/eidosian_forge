import configparser
import logging
import warnings
from configparser import ConfigParser
from pathlib import Path
from project import ProjectData
from .commands import run_qmlimportscanner
from . import DEFAULT_APP_ICON
def _find_and_set_project_dir(self):
    self.project_dir = self.source_file.parent
    self.set_value('app', 'input_file', str(self.source_file.relative_to(self.project_dir)))
    if self.project_dir != Path.cwd():
        self.set_value('app', 'project_dir', str(self.project_dir))
    else:
        self.set_value('app', 'project_dir', str(self.project_dir.relative_to(Path.cwd())))