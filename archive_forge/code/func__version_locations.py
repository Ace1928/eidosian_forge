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
@util.memoized_property
def _version_locations(self) -> Sequence[str]:
    if self.version_locations:
        return [os.path.abspath(util.coerce_resource_to_filename(location)) for location in self.version_locations]
    else:
        return (os.path.abspath(os.path.join(self.dir, 'versions')),)