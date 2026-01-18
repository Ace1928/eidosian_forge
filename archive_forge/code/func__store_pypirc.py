import os
from configparser import RawConfigParser
import warnings
from distutils.cmd import Command
def _store_pypirc(self, username, password):
    """Creates a default .pypirc file."""
    rc = self._get_rc_file()
    with os.fdopen(os.open(rc, os.O_CREAT | os.O_WRONLY, 384), 'w') as f:
        f.write(DEFAULT_PYPIRC % (username, password))