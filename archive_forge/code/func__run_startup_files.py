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
def _run_startup_files(self):
    """Run files from profile startup directory"""
    startup_dirs = [self.profile_dir.startup_dir] + [os.path.join(p, 'startup') for p in chain(ENV_CONFIG_DIRS, SYSTEM_CONFIG_DIRS)]
    startup_files = []
    if self.exec_PYTHONSTARTUP and os.environ.get('PYTHONSTARTUP', False) and (not (self.file_to_run or self.code_to_run or self.module_to_run)):
        python_startup = os.environ['PYTHONSTARTUP']
        self.log.debug('Running PYTHONSTARTUP file %s...', python_startup)
        try:
            self._exec_file(python_startup)
        except:
            self.log.warning('Unknown error in handling PYTHONSTARTUP file %s:', python_startup)
            self.shell.showtraceback()
    for startup_dir in startup_dirs[::-1]:
        startup_files += glob.glob(os.path.join(startup_dir, '*.py'))
        startup_files += glob.glob(os.path.join(startup_dir, '*.ipy'))
    if not startup_files:
        return
    self.log.debug('Running startup files from %s...', startup_dir)
    try:
        for fname in sorted(startup_files):
            self._exec_file(fname)
    except:
        self.log.warning('Unknown error in handling startup files:')
        self.shell.showtraceback()