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
def _bugs_properties_handler(revision):
    fixed_bug_urls = []
    related_bug_urls = []
    for bug_url, status in revision.iter_bugs():
        if status == 'fixed':
            fixed_bug_urls.append(bug_url)
        elif status == 'related':
            related_bug_urls.append(bug_url)
    ret = {}
    if fixed_bug_urls:
        text = ngettext('fixes bug', 'fixes bugs', len(fixed_bug_urls))
        ret[text] = ' '.join(fixed_bug_urls)
    if related_bug_urls:
        text = ngettext('related bug', 'related bugs', len(related_bug_urls))
        ret[text] = ' '.join(related_bug_urls)
    return ret