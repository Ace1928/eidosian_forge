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
def _compute_revno_str(branch, rev_id):
    """Compute the revno string from a rev_id.

    :return: The revno string, or None if the revision is not in the supplied
        branch.
    """
    try:
        revno = branch.revision_id_to_dotted_revno(rev_id)
    except errors.NoSuchRevision:
        return None
    else:
        return '.'.join((str(n) for n in revno))