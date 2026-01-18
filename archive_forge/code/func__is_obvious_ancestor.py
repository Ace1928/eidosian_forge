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
def _is_obvious_ancestor(branch, start_rev_id, end_rev_id):
    """Is start_rev_id an obvious ancestor of end_rev_id?"""
    if start_rev_id and end_rev_id:
        try:
            start_dotted = branch.revision_id_to_dotted_revno(start_rev_id)
            end_dotted = branch.revision_id_to_dotted_revno(end_rev_id)
        except errors.NoSuchRevision:
            return False
        if len(start_dotted) == 1 and len(end_dotted) == 1:
            return start_dotted[0] <= end_dotted[0]
        elif len(start_dotted) == 3 and len(end_dotted) == 3 and (start_dotted[0:1] == end_dotted[0:1]):
            return start_dotted[2] <= end_dotted[2]
        else:
            return False
    return True