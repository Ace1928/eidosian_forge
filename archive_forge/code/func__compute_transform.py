import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _compute_transform(self):
    with ui.ui_factory.nested_progress_bar() as child_pb:
        entries = self._entries_to_incorporate()
        entries = list(entries)
        for num, (entry, parent_id, relpath) in enumerate(entries):
            child_pb.update(gettext('Preparing file merge'), num, len(entries))
            parent_trans_id = self.tt.trans_id_file_id(parent_id)
            path = osutils.pathjoin(self._source_subpath, relpath)
            trans_id = transform.new_by_entry(path, self.tt, entry, parent_trans_id, self.other_tree)
    self._finish_computing_transform()