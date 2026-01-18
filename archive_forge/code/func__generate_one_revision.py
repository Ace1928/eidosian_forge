import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _generate_one_revision(branch, rev_id, br_rev_id, br_revno):
    if rev_id == br_rev_id:
        return [(br_rev_id, br_revno, 0)]
    else:
        revno_str = _compute_revno_str(branch, rev_id)
        return [(rev_id, revno_str, 0)]