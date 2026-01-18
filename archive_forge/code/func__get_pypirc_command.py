import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from . import DistlibException
from .compat import (HTTPBasicAuthHandler, Request, HTTPPasswordMgr,
from .util import zip_dir, ServerProxy
def _get_pypirc_command(self):
    """
        Get the distutils command for interacting with PyPI configurations.
        :return: the command.
        """
    from .util import _get_pypirc_command as cmd
    return cmd()