from ....conflicts import ConflictList
from ....errors import ConflictsInTree, UnknownFormatError
from ....graph import DictParentsProvider, Graph
from ....revision import NULL_REVISION
from ....tests import TestCase, TestCaseWithTransport
from ....tests.matchers import RevisionHistoryMatches
from ....transport import NoSuchFile
from ..rebase import (REBASE_CURRENT_REVID_FILENAME, REBASE_PLAN_FILENAME,
class RebaseTodoTests(TestCase):

    def test_done(self):

        class Repository:

            def has_revision(self, revid):
                return revid == 'bloe'
        self.assertEqual([], list(rebase_todo(Repository(), {'bla': ('bloe', [])})))

    def test_notstarted(self):

        class Repository:

            def has_revision(self, revid):
                return False
        self.assertEqual(['bla'], list(rebase_todo(Repository(), {'bla': ('bloe', [])})))

    def test_halfway(self):

        class Repository:

            def has_revision(self, revid):
                return revid == 'bloe'
        self.assertEqual(['ha'], list(rebase_todo(Repository(), {'bla': ('bloe', []), 'ha': ('hee', [])})))