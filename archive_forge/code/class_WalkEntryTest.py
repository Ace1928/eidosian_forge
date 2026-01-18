from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
class WalkEntryTest(TestCase):

    def setUp(self):
        super().setUp()
        self.store = MemoryObjectStore()

    def make_commits(self, commit_spec, **kwargs):
        times = kwargs.pop('times', [])
        attrs = kwargs.pop('attrs', {})
        for i, t in enumerate(times):
            attrs.setdefault(i + 1, {})['commit_time'] = t
        return build_commit_graph(self.store, commit_spec, attrs=attrs, **kwargs)

    def make_linear_commits(self, num_commits, **kwargs):
        commit_spec = []
        for i in range(1, num_commits + 1):
            c = [i]
            if i > 1:
                c.append(i - 1)
            commit_spec.append(c)
        return self.make_commits(commit_spec, **kwargs)

    def test_all_changes(self):
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        c1 = self.make_linear_commits(1, trees={1: [(b'x/a', blob_a), (b'y/b', blob_b)]})[0]
        walker = Walker(self.store, c1.id)
        walker_entry = next(iter(walker))
        changes = walker_entry.changes()
        entry_a = (b'x/a', F, blob_a.id)
        entry_b = (b'y/b', F, blob_b.id)
        self.assertEqual([TreeChange.add(entry_a), TreeChange.add(entry_b)], changes)

    def test_all_with_merge(self):
        blob_a = make_object(Blob, data=b'a')
        blob_a2 = make_object(Blob, data=b'a2')
        blob_b = make_object(Blob, data=b'b')
        blob_b2 = make_object(Blob, data=b'b2')
        x1, y2, m3 = self.make_commits([[1], [2], [3, 1, 2]], trees={1: [(b'x/a', blob_a)], 2: [(b'y/b', blob_b)], 3: [(b'x/a', blob_a2), (b'y/b', blob_b2)]})
        walker = Walker(self.store, m3.id)
        entries = list(walker)
        walker_entry = entries[0]
        self.assertEqual(walker_entry.commit.id, m3.id)
        changes = walker_entry.changes()
        self.assertEqual(2, len(changes))
        entry_a = (b'x/a', F, blob_a.id)
        entry_a2 = (b'x/a', F, blob_a2.id)
        entry_b = (b'y/b', F, blob_b.id)
        entry_b2 = (b'y/b', F, blob_b2.id)
        self.assertEqual([[TreeChange(CHANGE_MODIFY, entry_a, entry_a2), TreeChange.add(entry_a2)], [TreeChange.add(entry_b2), TreeChange(CHANGE_MODIFY, entry_b, entry_b2)]], changes)

    def test_filter_changes(self):
        blob_a = make_object(Blob, data=b'a')
        blob_b = make_object(Blob, data=b'b')
        c1 = self.make_linear_commits(1, trees={1: [(b'x/a', blob_a), (b'y/b', blob_b)]})[0]
        walker = Walker(self.store, c1.id)
        walker_entry = next(iter(walker))
        changes = walker_entry.changes(path_prefix=b'x')
        entry_a = (b'a', F, blob_a.id)
        self.assertEqual([TreeChange.add(entry_a)], changes)

    def test_filter_with_merge(self):
        blob_a = make_object(Blob, data=b'a')
        blob_a2 = make_object(Blob, data=b'a2')
        blob_b = make_object(Blob, data=b'b')
        blob_b2 = make_object(Blob, data=b'b2')
        x1, y2, m3 = self.make_commits([[1], [2], [3, 1, 2]], trees={1: [(b'x/a', blob_a)], 2: [(b'y/b', blob_b)], 3: [(b'x/a', blob_a2), (b'y/b', blob_b2)]})
        walker = Walker(self.store, m3.id)
        entries = list(walker)
        walker_entry = entries[0]
        self.assertEqual(walker_entry.commit.id, m3.id)
        changes = walker_entry.changes(b'x')
        self.assertEqual(1, len(changes))
        entry_a = (b'a', F, blob_a.id)
        entry_a2 = (b'a', F, blob_a2.id)
        self.assertEqual([[TreeChange(CHANGE_MODIFY, entry_a, entry_a2)]], changes)