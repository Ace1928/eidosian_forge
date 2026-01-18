from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
@classmethod
def _from_string(cls, string: str) -> 'Actor':
    """Create an Actor from a string.

        :param string: The string, which is expected to be in regular git format::

            John Doe <jdoe@example.com>

        :return: Actor
        """
    m = cls.name_email_regex.search(string)
    if m:
        name, email = m.groups()
        return Actor(name, email)
    else:
        m = cls.name_only_regex.search(string)
        if m:
            return Actor(m.group(1), None)
        return Actor(string, None)