import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
from zipfile import ZipFile
from jinja2 import Environment, FileSystemLoader
from .. import run_command
@dataclass
class AndroidData:
    """
    Dataclass to store all the Android data obtained through cli
    """
    wheel_pyside: Path
    wheel_shiboken: Path
    ndk_path: Path
    sdk_path: Path