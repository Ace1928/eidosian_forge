import base64
import dataclasses
import datetime
import errno
import json
import os
import subprocess
import tempfile
import time
import typing
from typing import Optional
from tensorboard import version
from tensorboard.util import tb_logging
@dataclasses.dataclass(frozen=True)
class StartFailed:
    """Possible return value of the `start` function.

    Indicates that a call to `start` tried to launch a new TensorBoard
    instance, but the subprocess exited with the given exit code and
    output streams. (If the contents of the output streams are no longer
    available---e.g., because the user has emptied /tmp/---then the
    corresponding values will be `None`.)

    Attributes:
      exit_code: As `Popen.returncode` (negative for signal).
      stdout: Error message to stdout if the stream could not be read.
      stderr: Error message to stderr if the stream could not be read.
    """
    exit_code: int
    stdout: Optional[str]
    stderr: Optional[str]