import abc
import atexit
import contextlib
import logging
import os
import pathlib
import random
import tempfile
import time
import typing
import warnings
from . import constants, exceptions, portalocker
def _prepare_fh(self, fh: typing.IO) -> typing.IO:
    """
        Prepare the filehandle for usage

        If truncate is a number, the file will be truncated to that amount of
        bytes
        """
    if self.truncate:
        fh.seek(0)
        fh.truncate(0)
    return fh