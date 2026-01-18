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
class StartLaunched:
    """Possible return value of the `start` function.

    Indicates that a call to `start` successfully launched a new
    TensorBoard process, which is available with the provided info.

    Attributes:
      info: A `TensorBoardInfo` object.
    """
    info: TensorBoardInfo