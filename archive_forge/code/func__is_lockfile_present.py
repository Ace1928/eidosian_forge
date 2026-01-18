from __future__ import (absolute_import, division, print_function)
import os
import time
import glob
from abc import ABCMeta, abstractmethod
from ansible.module_utils.six import with_metaclass
def _is_lockfile_present(self):
    return (os.path.isfile(self.lockfile) or glob.glob(self.lockfile)) and self.is_lockfile_pid_valid()