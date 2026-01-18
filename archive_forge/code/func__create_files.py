import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def _create_files(tt, tree, desired_files, pb, offset, accelerator_tree, hardlink):
    total = len(desired_files) + offset
    wt = tt._tree
    if accelerator_tree is None:
        new_desired_files = desired_files
    else:
        iter = accelerator_tree.iter_changes(tree, include_unchanged=True)
        unchanged = [change.path for change in iter if not (change.changed_content or change.executable[0] != change.executable[1])]
        if accelerator_tree.supports_content_filtering():
            unchanged = [(tp, ap) for tp, ap in unchanged if not next(accelerator_tree.iter_search_rules([ap]))]
        unchanged = dict(unchanged)
        new_desired_files = []
        count = 0
        for unused_tree_path, (trans_id, tree_path, text_sha1) in desired_files:
            accelerator_path = unchanged.get(tree_path)
            if accelerator_path is None:
                new_desired_files.append((tree_path, (trans_id, tree_path, text_sha1)))
                continue
            pb.update(gettext('Adding file contents'), count + offset, total)
            if hardlink:
                tt.create_hardlink(accelerator_tree.abspath(accelerator_path), trans_id)
            else:
                with accelerator_tree.get_file(accelerator_path) as f:
                    chunks = osutils.file_iterator(f)
                    if wt.supports_content_filtering():
                        filters = wt._content_filter_stack(tree_path)
                        chunks = filtered_output_bytes(chunks, filters, ContentFilterContext(tree_path, tree))
                    tt.create_file(chunks, trans_id, sha1=text_sha1)
            count += 1
        offset += count
    for count, ((trans_id, tree_path, text_sha1), contents) in enumerate(tree.iter_files_bytes(new_desired_files)):
        if wt.supports_content_filtering():
            filters = wt._content_filter_stack(tree_path)
            contents = filtered_output_bytes(contents, filters, ContentFilterContext(tree_path, tree))
        tt.create_file(contents, trans_id, sha1=text_sha1)
        pb.update(gettext('Adding file contents'), count + offset, total)