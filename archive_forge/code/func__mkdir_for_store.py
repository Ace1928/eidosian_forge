import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.stores import ByteStore
from langchain.storage.exceptions import InvalidKeyException
def _mkdir_for_store(self, dir: Path) -> None:
    """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
    if not dir.exists():
        self._mkdir_for_store(dir.parent)
        dir.mkdir(exist_ok=True)
        if self.chmod_dir is not None:
            os.chmod(dir, self.chmod_dir)