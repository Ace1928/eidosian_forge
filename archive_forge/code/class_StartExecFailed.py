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
class StartExecFailed:
    """Possible return value of the `start` function.

    Indicates that a call to `start` failed to invoke the subprocess.

    Attributes:
      os_error: `OSError` due to `Popen` invocation.
      explicit_binary: If the TensorBoard executable was chosen via the
        `TENSORBOARD_BINARY` environment variable, then this field contains
        the path to that binary; otherwise `None`.
    """
    os_error: OSError
    explicit_binary: Optional[str]