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
def _match_filter(searchRE, rev):
    strings = {'message': (rev.message,), 'committer': (rev.committer,), 'author': rev.get_apparent_authors(), 'bugs': list(rev.iter_bugs())}
    strings[''] = [item for inner_list in strings.values() for item in inner_list]
    for k, v in searchRE:
        if k in strings and (not _match_any_filter(strings[k], v)):
            return False
    return True