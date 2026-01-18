import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
class PkgInstall:
    win: str = 'choco install [flags]'
    mac: str = 'brew [flags] install '
    linux: str = 'apt-get -y [flags] install'

    @classmethod
    def get_args(cls, binary: str, flags: List[str]=None):
        """
        Builds the install args for the system
        """
        flag_str = ' '.join(flags) if flags else ''
        if sys.platform.startswith('win'):
            b = f'{cls.win} {binary}'
            b = b.replace('[flags]', flag_str)
        if sys.platform.startswith('linux'):
            b = f'{cls.linux} {binary}'
            b = b.replace('[flags]', flag_str)
        if sys.platform.startswith('darwin'):
            b = f'{cls.mac} {binary}'
            b = b.replace('[flags]', flag_str)
        return shlex.split(b)