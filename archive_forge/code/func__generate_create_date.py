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
def _generate_create_date(self) -> datetime.datetime:
    if self.timezone is not None:
        if ZoneInfo is None:
            raise util.CommandError("Python >= 3.9 is required for timezone support orthe 'backports.zoneinfo' package must be installed.")
        try:
            tzinfo = ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError:
            tzinfo = None
        if tzinfo is None:
            try:
                tzinfo = ZoneInfo(self.timezone.upper())
            except ZoneInfoNotFoundError:
                raise util.CommandError("Can't locate timezone: %s" % self.timezone) from None
        create_date = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).astimezone(tzinfo)
    else:
        create_date = datetime.datetime.now()
    return create_date