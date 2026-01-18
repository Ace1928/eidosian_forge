import configparser
import os
import shlex
import subprocess
from os.path import expanduser, expandvars
from pathlib import Path
from typing import List, Optional, Union
from gitlab.const import USER_AGENT
def _resolve_file(filepath: Union[Path, str]) -> str:
    resolved = Path(filepath).resolve(strict=True)
    return str(resolved)