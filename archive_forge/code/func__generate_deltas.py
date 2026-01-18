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
def _generate_deltas(repository, log_rev_iterator, delta_type, files, direction):
    """Create deltas for each batch of revisions in log_rev_iterator.

    If we're only generating deltas for the sake of filtering against
    files, we stop generating deltas once all files reach the
    appropriate life-cycle point. If we're receiving data newest to
    oldest, then that life-cycle point is 'add', otherwise it's 'remove'.
    """
    check_files = files is not None and len(files) > 0
    if check_files:
        file_set = set(files)
        if direction == 'reverse':
            stop_on = 'add'
        else:
            stop_on = 'remove'
    else:
        file_set = None
    for revs in log_rev_iterator:
        if check_files and (not file_set):
            return
        revisions = [rev[1] for rev in revs]
        new_revs = []
        if delta_type == 'full' and (not check_files):
            deltas = repository.get_revision_deltas(revisions)
            for rev, delta in zip(revs, deltas):
                new_revs.append((rev[0], rev[1], delta))
        else:
            deltas = repository.get_revision_deltas(revisions, specific_files=file_set)
            for rev, delta in zip(revs, deltas):
                if check_files:
                    if delta is None or not delta.has_changed():
                        continue
                    else:
                        _update_files(delta, file_set, stop_on)
                        if delta_type is None:
                            delta = None
                        elif delta_type == 'full':
                            rev_id = rev[0][0]
                            delta = repository.get_revision_delta(rev_id)
                new_revs.append((rev[0], rev[1], delta))
        yield new_revs