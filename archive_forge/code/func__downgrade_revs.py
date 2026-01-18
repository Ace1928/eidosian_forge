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
def _downgrade_revs(self, destination: str, current_rev: Optional[str]) -> List[RevisionStep]:
    with self._catch_revision_errors(ancestor='Destination %(end)s is not a valid downgrade target from current head(s)', end=destination):
        revs = self.iterate_revisions(current_rev, destination, select_for_downgrade=True)
        return [migration.MigrationStep.downgrade_from_script(self.revision_map, script) for script in revs]