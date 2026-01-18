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
@contextmanager
def _catch_revision_errors(self, ancestor: Optional[str]=None, multiple_heads: Optional[str]=None, start: Optional[str]=None, end: Optional[str]=None, resolution: Optional[str]=None) -> Iterator[None]:
    try:
        yield
    except revision.RangeNotAncestorError as rna:
        if start is None:
            start = cast(Any, rna.lower)
        if end is None:
            end = cast(Any, rna.upper)
        if not ancestor:
            ancestor = 'Requested range %(start)s:%(end)s does not refer to ancestor/descendant revisions along the same branch'
        ancestor = ancestor % {'start': start, 'end': end}
        raise util.CommandError(ancestor) from rna
    except revision.MultipleHeads as mh:
        if not multiple_heads:
            multiple_heads = "Multiple head revisions are present for given argument '%(head_arg)s'; please specify a specific target revision, '<branchname>@%(head_arg)s' to narrow to a specific head, or 'heads' for all heads"
        multiple_heads = multiple_heads % {'head_arg': end or mh.argument, 'heads': util.format_as_comma(mh.heads)}
        raise util.CommandError(multiple_heads) from mh
    except revision.ResolutionError as re:
        if resolution is None:
            resolution = "Can't locate revision identified by '%s'" % re.argument
        raise util.CommandError(resolution) from re
    except revision.RevisionError as err:
        raise util.CommandError(err.args[0]) from err