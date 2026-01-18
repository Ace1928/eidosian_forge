import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
def _revision_range_to_revid_range(revision_range):
    rev_id1 = None
    rev_id2 = None
    if revision_range[0] is not None:
        rev_id1 = revision_range[0].rev_id
    if revision_range[1] is not None:
        rev_id2 = revision_range[1].rev_id
    return (rev_id1, rev_id2)