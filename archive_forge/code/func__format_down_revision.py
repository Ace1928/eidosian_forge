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
def _format_down_revision(self) -> str:
    if not self.down_revision:
        return '<base>'
    else:
        return util.format_as_comma(self._versioned_down_revisions)