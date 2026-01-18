import os
import stat
import sys
import time
import warnings
from io import BytesIO
from typing import (
from .errors import (
from .file import GitFile
from .hooks import (
from .line_ending import BlobNormalizer, TreeBlobNormalizer
from .object_store import (
from .objects import (
from .pack import generate_unpacked_objects
from .refs import (
def _add_graftpoints(self, updated_graftpoints: Dict[bytes, List[bytes]]):
    """Add or modify graftpoints.

        Args:
          updated_graftpoints: Dict of commit shas to list of parent shas
        """
    for commit, parents in updated_graftpoints.items():
        for sha in [commit, *parents]:
            check_hexsha(sha, 'Invalid graftpoint')
    self._graftpoints.update(updated_graftpoints)