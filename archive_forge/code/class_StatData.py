import dataclasses
import glob as py_glob
import io
import os
import os.path
import sys
import tempfile
from tensorboard.compat.tensorflow_stub import compat, errors
@dataclasses.dataclass(frozen=True)
class StatData:
    """Data returned from the Stat call.

    Attributes:
      length: Length of the data content.
    """
    length: int