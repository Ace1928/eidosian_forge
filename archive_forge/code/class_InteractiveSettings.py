import argparse
import json
import os
import platform
import re
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union
from tqdm.auto import tqdm
from cmdstanpy import _DOT_CMDSTAN
from cmdstanpy.utils import (
from cmdstanpy.utils.cmdstan import get_download_url
from . import progress as progbar
class InteractiveSettings:
    """
    Installation settings provided on-demand in an interactive format.

    This provides the same set of properties as the ``InstallationSettings``
    object, but rather than them being fixed by the constructor the user is
    asked for input whenever they are accessed for the first time.
    """

    @cached_property
    def version(self) -> str:
        latest = latest_version()
        print('Which version would you like to install?')
        print(f'Default: {latest}')
        answer = input('Type version or hit enter to continue: ')
        return answer if answer else latest

    @cached_property
    def dir(self) -> str:
        directory = home_cmdstan()
        print('Where would you like to install CmdStan?')
        print(f'Default: {directory}')
        answer = input('Type full path or hit enter to continue: ')
        return os.path.expanduser(answer) if answer else directory

    @cached_property
    def progress(self) -> bool:
        print('Show installation progress bars?')
        print('Default: y')
        answer = input('[y/n]: ')
        return yes_no(answer, True)

    @cached_property
    def verbose(self) -> bool:
        print('Show verbose output of the installation process?')
        print('Default: n')
        answer = input('[y/n]: ')
        return yes_no(answer, False)

    @cached_property
    def overwrite(self) -> bool:
        print('Overwrite existing CmdStan installation?')
        print('Default: n')
        answer = input('[y/n]: ')
        return yes_no(answer, False)

    @cached_property
    def compiler(self) -> bool:
        if not is_windows():
            return False
        print('Would you like to install the RTools40 C++ toolchain?')
        print('A C++ toolchain is required for CmdStan.')
        print("If you are not sure if you need the toolchain or not, the most likely case is you do need it, and should answer 'y'.")
        print('Default: n')
        answer = input('[y/n]: ')
        return yes_no(answer, False)

    @cached_property
    def cores(self) -> int:
        max_cpus = os.cpu_count() or 1
        print('How many CPU cores would you like to use for installing and compiling CmdStan?')
        print(f'Default: 1, Max: {max_cpus}')
        answer = input('Enter a number or hit enter to continue: ')
        try:
            return min(max_cpus, max(int(answer), 1))
        except ValueError:
            return 1