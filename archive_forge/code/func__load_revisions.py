from __future__ import annotations
from contextlib import contextmanager
import datetime
import os
import re
import shutil
import sys
from types import ModuleType
from typing import Any
from typing import cast
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import revision
from . import write_hooks
from .. import util
from ..runtime import migration
from ..util import compat
from ..util import not_none
def _load_revisions(self) -> Iterator[Script]:
    if self.version_locations:
        paths = [vers for vers in self._version_locations if os.path.exists(vers)]
    else:
        paths = [self.versions]
    dupes = set()
    for vers in paths:
        for file_path in Script._list_py_dir(self, vers):
            real_path = os.path.realpath(file_path)
            if real_path in dupes:
                util.warn('File %s loaded twice! ignoring. Please ensure version_locations is unique.' % real_path)
                continue
            dupes.add(real_path)
            filename = os.path.basename(real_path)
            dir_name = os.path.dirname(real_path)
            script = Script._from_filename(self, dir_name, filename)
            if script is None:
                continue
            yield script