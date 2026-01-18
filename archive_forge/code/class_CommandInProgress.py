import atexit
import os
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, TypedDict, Union
from urllib.parse import urlparse
from huggingface_hub.constants import REPO_TYPES_URL_PREFIXES, REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from .hf_api import HfApi, repo_type_and_id_from_hf_id
from .lfs import LFS_MULTIPART_UPLOAD_COMMAND
from .utils import (
from .utils._deprecation import _deprecate_method
class CommandInProgress:
    """
    Utility to follow commands launched asynchronously.
    """

    def __init__(self, title: str, is_done_method: Callable, status_method: Callable, process: subprocess.Popen, post_method: Optional[Callable]=None):
        self.title = title
        self._is_done = is_done_method
        self._status = status_method
        self._process = process
        self._stderr = ''
        self._stdout = ''
        self._post_method = post_method

    @property
    def is_done(self) -> bool:
        """
        Whether the process is done.
        """
        result = self._is_done()
        if result and self._post_method is not None:
            self._post_method()
            self._post_method = None
        return result

    @property
    def status(self) -> int:
        """
        The exit code/status of the current action. Will return `0` if the
        command has completed successfully, and a number between 1 and 255 if
        the process errored-out.

        Will return -1 if the command is still ongoing.
        """
        return self._status()

    @property
    def failed(self) -> bool:
        """
        Whether the process errored-out.
        """
        return self.status > 0

    @property
    def stderr(self) -> str:
        """
        The current output message on the standard error.
        """
        if self._process.stderr is not None:
            self._stderr += self._process.stderr.read()
        return self._stderr

    @property
    def stdout(self) -> str:
        """
        The current output message on the standard output.
        """
        if self._process.stdout is not None:
            self._stdout += self._process.stdout.read()
        return self._stdout

    def __repr__(self):
        status = self.status
        if status == -1:
            status = 'running'
        return f'[{self.title} command, status code: {status}, {('in progress.' if not self.is_done else 'finished.')} PID: {self._process.pid}]'