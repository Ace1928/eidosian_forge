import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def _run_exec_files(self):
    """Run files from IPythonApp.exec_files"""
    if not self.exec_files:
        return
    self.log.debug('Running files in IPythonApp.exec_files...')
    try:
        for fname in self.exec_files:
            self._exec_file(fname)
    except:
        self.log.warning('Unknown error in handling IPythonApp.exec_files:')
        self.shell.showtraceback()