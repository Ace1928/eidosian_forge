import os
import os.path
from ._paml import Paml
from . import _parse_baseml
class BasemlError(EnvironmentError):
    """BASEML failed. Run with verbose=True to view BASEML's error message."""