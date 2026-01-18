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
def _rev_path(self, path: str, rev_id: str, message: Optional[str], create_date: datetime.datetime) -> str:
    epoch = int(create_date.timestamp())
    slug = '_'.join(_slug_re.findall(message or '')).lower()
    if len(slug) > self.truncate_slug_length:
        slug = slug[:self.truncate_slug_length].rsplit('_', 1)[0] + '_'
    filename = '%s.py' % (self.file_template % {'rev': rev_id, 'slug': slug, 'epoch': epoch, 'year': create_date.year, 'month': create_date.month, 'day': create_date.day, 'hour': create_date.hour, 'minute': create_date.minute, 'second': create_date.second})
    return os.path.join(path, filename)