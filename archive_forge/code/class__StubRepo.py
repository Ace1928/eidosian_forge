import errno
import os
import shutil
import socket
import tempfile
from ...objects import hex_to_sha
from ...protocol import CAPABILITY_SIDE_BAND_64K
from ...repo import Repo
from ...server import ReceivePackHandler
from ..utils import tear_down_repo
from .utils import require_git_version, run_git_or_fail
class _StubRepo:
    """A stub repo that just contains a path to tear down."""

    def __init__(self, name) -> None:
        temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(temp_dir, name)
        os.mkdir(self.path)

    def close(self):
        pass