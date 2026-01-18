from __future__ import annotations
from pathlib import Path
from collections import deque
from contextlib import suppress
from copy import deepcopy
from fnmatch import fnmatch
import argparse
import asyncio
import datetime
import enum
import json
import multiprocessing
import os
import pickle
import platform
import random
import re
import signal
import subprocess
import shlex
import sys
import textwrap
import time
import typing as T
import unicodedata
import xml.etree.ElementTree as et
from . import build
from . import environment
from . import mlog
from .coredata import MesonVersionMismatchException, major_versions_differ
from .coredata import version as coredata_version
from .mesonlib import (MesonException, OptionKey, OrderedSet, RealPathAction,
from .mintro import get_infodir, load_info_file
from .programs import ExternalProgram
from .backend.backends import TestProtocol, TestSerialisation
def _get_test_cmd(self) -> T.Optional[T.List[str]]:
    testentry = self.test.fname[0]
    if self.options.no_rebuild and self.test.cmd_is_built and (not os.path.isfile(testentry)):
        raise TestException(f'The test program {testentry!r} does not exist. Cannot run tests before building them.')
    if testentry.endswith('.jar'):
        return ['java', '-jar'] + self.test.fname
    elif not self.test.is_cross_built and run_with_mono(testentry):
        return ['mono'] + self.test.fname
    elif self.test.cmd_is_exe and self.test.is_cross_built and self.test.needs_exe_wrapper:
        if self.test.exe_wrapper is None:
            return None
        elif self.test.cmd_is_exe:
            if not self.test.exe_wrapper.found():
                msg = 'The exe_wrapper defined in the cross file {!r} was not found. Please check the command and/or add it to PATH.'
                raise TestException(msg.format(self.test.exe_wrapper.name))
            return self.test.exe_wrapper.get_command() + self.test.fname
    elif self.test.cmd_is_built and (not self.test.cmd_is_exe) and is_windows():
        test_cmd = ExternalProgram._shebang_to_cmd(self.test.fname[0])
        if test_cmd is not None:
            test_cmd += self.test.fname[1:]
        return test_cmd
    return self.test.fname