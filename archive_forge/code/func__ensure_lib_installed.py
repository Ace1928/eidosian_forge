import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def _ensure_lib_installed(cls, library: str, pip_name: str=None, upgrade: bool=False):
    clean_lib = cls.get_requirement(library, True)
    if not cls.is_available(clean_lib):
        cls.install_library(pip_name or library, upgrade=upgrade)