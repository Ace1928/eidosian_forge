import errno
import logging
import os
import subprocess
import typing
from abc import ABC
from abc import abstractmethod
from io import IOBase
from platform import system
from subprocess import PIPE
from time import sleep
from urllib import request
from urllib.error import URLError
from selenium.common.exceptions import WebDriverException
from selenium.types import SubprocessStdAlias
from selenium.webdriver.common import utils
def _terminate_process(self) -> None:
    """Terminate the child process.

        On POSIX this attempts a graceful SIGTERM followed by a SIGKILL,
        on a Windows OS kill is an alias to terminate.  Terminating does
        not raise itself if something has gone wrong but (currently)
        silently ignores errors here.
        """
    try:
        stdin, stdout, stderr = (self.process.stdin, self.process.stdout, self.process.stderr)
        for stream in (stdin, stdout, stderr):
            try:
                stream.close()
            except AttributeError:
                pass
        self.process.terminate()
        try:
            self.process.wait(60)
        except subprocess.TimeoutExpired:
            logger.error('Service process refused to terminate gracefully with SIGTERM, escalating to SIGKILL.', exc_info=True)
            self.process.kill()
    except OSError:
        logger.error('Error terminating service process.', exc_info=True)