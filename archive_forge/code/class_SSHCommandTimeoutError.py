import os
import re
import time
import logging
import warnings
import subprocess
from typing import List, Type, Tuple, Union, Optional, cast
from os.path import join as pjoin
from os.path import split as psplit
from libcloud.utils.py3 import StringIO, b
from libcloud.utils.logging import ExtraLogFormatter
class SSHCommandTimeoutError(Exception):
    """
    Exception which is raised when an SSH command times out.
    """

    def __init__(self, cmd, timeout, stdout=None, stderr=None):
        self.cmd = cmd
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr
        self.message = "Command didn't finish in %s seconds" % timeout
        super().__init__(self.message)

    def __repr__(self):
        return '<SSHCommandTimeoutError: cmd="{}",timeout={})>'.format(self.cmd, self.timeout)

    def __str__(self):
        return self.__repr__()