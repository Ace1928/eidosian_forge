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
def _filter_re(searchRE, log_rev_iterator):
    for revs in log_rev_iterator:
        new_revs = [rev for rev in revs if _match_filter(searchRE, rev[1])]
        if new_revs:
            yield new_revs