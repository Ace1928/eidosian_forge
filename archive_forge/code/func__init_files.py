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
def _init_files(self, bare: bool, symlinks: Optional[bool]=None) -> None:
    """Initialize a default set of named files."""
    from .config import ConfigFile
    self._put_named_file('description', b'Unnamed repository')
    f = BytesIO()
    cf = ConfigFile()
    cf.set('core', 'repositoryformatversion', '0')
    if self._determine_file_mode():
        cf.set('core', 'filemode', True)
    else:
        cf.set('core', 'filemode', False)
    if symlinks is None and (not bare):
        symlinks = self._determine_symlinks()
    if symlinks is False:
        cf.set('core', 'symlinks', symlinks)
    cf.set('core', 'bare', bare)
    cf.set('core', 'logallrefupdates', True)
    cf.write_to_file(f)
    self._put_named_file('config', f.getvalue())
    self._put_named_file(os.path.join('info', 'exclude'), b'')