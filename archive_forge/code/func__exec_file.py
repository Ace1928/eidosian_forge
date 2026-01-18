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
def _exec_file(self, fname, shell_futures=False):
    try:
        full_filename = filefind(fname, [u'.', self.ipython_dir])
    except IOError:
        self.log.warning('File not found: %r' % fname)
        return
    save_argv = sys.argv
    sys.argv = [full_filename] + self.extra_args[1:]
    try:
        if os.path.isfile(full_filename):
            self.log.info('Running file in user namespace: %s' % full_filename)
            with preserve_keys(self.shell.user_ns, '__file__'):
                self.shell.user_ns['__file__'] = fname
                if full_filename.endswith('.ipy') or full_filename.endswith('.ipynb'):
                    self.shell.safe_execfile_ipy(full_filename, shell_futures=shell_futures)
                else:
                    self.shell.safe_execfile(full_filename, self.shell.user_ns, shell_futures=shell_futures, raise_exceptions=True)
    finally:
        sys.argv = save_argv