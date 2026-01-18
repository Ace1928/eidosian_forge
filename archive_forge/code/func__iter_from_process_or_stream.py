import datetime
import re
from subprocess import Popen, PIPE
from gitdb import IStream
from git.util import hex_to_bin, Actor, Stats, finalize_process
from git.diff import Diffable
from git.cmd import Git
from .tree import Tree
from . import base
from .util import (
from time import time, daylight, altzone, timezone, localtime
import os
from io import BytesIO
import logging
from collections import defaultdict
from typing import (
from git.types import PathLike, Literal
@classmethod
def _iter_from_process_or_stream(cls, repo: 'Repo', proc_or_stream: Union[Popen, IO]) -> Iterator['Commit']:
    """Parse out commit information into a list of Commit objects.

        We expect one-line per commit, and parse the actual commit information directly
        from our lighting fast object database.

        :param proc: git-rev-list process instance - one sha per line
        :return: iterator supplying :class:`Commit` objects
        """
    if hasattr(proc_or_stream, 'wait'):
        proc_or_stream = cast(Popen, proc_or_stream)
        if proc_or_stream.stdout is not None:
            stream = proc_or_stream.stdout
    elif hasattr(proc_or_stream, 'readline'):
        proc_or_stream = cast(IO, proc_or_stream)
        stream = proc_or_stream
    readline = stream.readline
    while True:
        line = readline()
        if not line:
            break
        hexsha = line.strip()
        if len(hexsha) > 40:
            hexsha, _ = line.split(None, 1)
        assert len(hexsha) == 40, 'Invalid line: %s' % hexsha
        yield cls(repo, hex_to_bin(hexsha))
    if hasattr(proc_or_stream, 'wait'):
        proc_or_stream = cast(Popen, proc_or_stream)
        finalize_process(proc_or_stream)