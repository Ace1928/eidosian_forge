import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
import PySide6 as ref_mod
def _extend_path_var(var, value, prepend=False):
    env_value = os.environ.get(var)
    if env_value:
        env_value = f'{value}{os.pathsep}{env_value}' if prepend else f'{env_value}{os.pathsep}{value}'
    else:
        env_value = value
    os.environ[var] = env_value