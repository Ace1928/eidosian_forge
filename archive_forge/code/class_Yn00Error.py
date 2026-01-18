import os.path
from ._paml import Paml
from . import _parse_yn00
class Yn00Error(EnvironmentError):
    """yn00 failed. Run with verbose=True to view yn00's error message."""