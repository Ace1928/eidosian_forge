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
class Diff3Merger(Merge3Merger):
    """Three-way merger using external diff3 for text merging"""
    requires_file_merge_plan = False

    def dump_file(self, temp_dir, name, tree, path):
        out_path = osutils.pathjoin(temp_dir, name)
        with open(out_path, 'wb') as out_file:
            in_file = tree.get_file(path)
            for line in in_file:
                out_file.write(line)
        return out_path

    def text_merge(self, trans_id, paths):
        """Perform a diff3 merge using a specified file-id and trans-id.
        If conflicts are encountered, .BASE, .THIS. and .OTHER conflict files
        will be dumped, and a will be conflict noted.
        """
        import breezy.patch
        base_path, other_path, this_path = paths
        with tempfile.TemporaryDirectory(prefix='bzr-') as temp_dir:
            new_file = osutils.pathjoin(temp_dir, 'new')
            this = self.dump_file(temp_dir, 'this', self.this_tree, this_path)
            base = self.dump_file(temp_dir, 'base', self.base_tree, base_path)
            other = self.dump_file(temp_dir, 'other', self.other_tree, other_path)
            status = breezy.patch.diff3(new_file, this, base, other)
            if status not in (0, 1):
                raise errors.BzrError('Unhandled diff3 exit code')
            with open(new_file, 'rb') as f:
                self.tt.create_file(f, trans_id)
            if status == 1:
                name = self.tt.final_name(trans_id)
                parent_id = self.tt.final_parent(trans_id)
                self._dump_conflicts(name, paths, parent_id)
                self._raw_conflicts.append(('text conflict', trans_id))