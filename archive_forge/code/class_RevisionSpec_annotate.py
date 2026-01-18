from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_annotate(RevisionIDSpec):
    prefix = 'annotate:'
    help_txt = 'Select the revision that last modified the specified line.\n\n    Select the revision that last modified the specified line.  Line is\n    specified as path:number.  Path is a relative path to the file.  Numbers\n    start at 1, and are relative to the current version, not the last-\n    committed version of the file.\n    '

    def _raise_invalid(self, numstring, context_branch):
        raise InvalidRevisionSpec(self.user_spec, context_branch, 'No such line: %s' % numstring)

    def _as_revision_id(self, context_branch):
        path, numstring = self.spec.rsplit(':', 1)
        try:
            index = int(numstring) - 1
        except ValueError:
            self._raise_invalid(numstring, context_branch)
        tree, file_path = workingtree.WorkingTree.open_containing(path)
        with tree.lock_read():
            if not tree.has_filename(file_path):
                raise InvalidRevisionSpec(self.user_spec, context_branch, "File '%s' is not versioned." % file_path)
            revision_ids = [r for r, l in tree.annotate_iter(file_path)]
        try:
            revision_id = revision_ids[index]
        except IndexError:
            self._raise_invalid(numstring, context_branch)
        if revision_id == revision.CURRENT_REVISION:
            raise InvalidRevisionSpec(self.user_spec, context_branch, 'Line %s has not been committed.' % numstring)
        return revision_id