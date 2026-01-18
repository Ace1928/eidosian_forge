from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def _check_weaves(self, storebar):
    storebar.update('text-index', 0, 2)
    if self.repository._format.fast_deltas:
        weave_checker = self.repository._get_versioned_file_checker(ancestors=self.ancestors)
    else:
        weave_checker = self.repository._get_versioned_file_checker(text_key_references=self.text_key_references, ancestors=self.ancestors)
    storebar.update('file-graph', 1)
    wrongs, unused_versions = weave_checker.check_file_version_parents(self.repository.texts)
    self.checked_weaves = weave_checker.file_ids
    for text_key, (stored_parents, correct_parents) in wrongs.items():
        weave_id = text_key[0]
        revision_id = text_key[-1]
        weave_parents = tuple([parent[-1] for parent in stored_parents])
        correct_parents = tuple([parent[-1] for parent in correct_parents])
        self.inconsistent_parents.append((revision_id, weave_id, weave_parents, correct_parents))
    self.unreferenced_versions.update(unused_versions)