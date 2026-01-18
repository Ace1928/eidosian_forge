from .. import ui
from ..branch import Branch
from ..check import Check
from ..i18n import gettext
from ..revision import NULL_REVISION
from ..trace import note
from ..workingtree import WorkingTree
def _check_revisions(self, revisions_iterator):
    """Check revision objects by decorating a generator.

        :param revisions_iterator: An iterator of(revid, Revision-or-None).
        :return: A generator of the contents of revisions_iterator.
        """
    self.planned_revisions = set()
    for revid, revision in revisions_iterator:
        yield (revid, revision)
        self._check_one_rev(revid, revision)
    self.planned_revisions = list(self.planned_revisions)