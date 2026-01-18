import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
class TestLCAMultiWay(tests.TestCase):

    def assertLCAMultiWay(self, expected, base, lcas, other, this, allow_overriding_lca=True):
        self.assertEqual(expected, _mod_merge.Merge3Merger._lca_multi_way((base, lcas), other, this, allow_overriding_lca=allow_overriding_lca))

    def test_other_equal_equal_lcas(self):
        """Test when OTHER=LCA and all LCAs are identical."""
        self.assertLCAMultiWay('this', 'bval', ['bval', 'bval'], 'bval', 'bval')
        self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval'], 'lcaval', 'bval')
        self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval', 'lcaval'], 'lcaval', 'bval')
        self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval', 'lcaval'], 'lcaval', 'tval')
        self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval', 'lcaval'], 'lcaval', None)

    def test_other_equal_this(self):
        """Test when other and this are identical."""
        self.assertLCAMultiWay('this', 'bval', ['bval', 'bval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', 'bval', ['lcaval', 'lcaval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', 'bval', ['cval', 'dval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', 'bval', [None, 'lcaval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', None, [None, 'lcaval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', None, ['lcaval', 'lcaval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', None, ['cval', 'dval'], 'oval', 'oval')
        self.assertLCAMultiWay('this', None, ['cval', 'dval'], None, None)
        self.assertLCAMultiWay('this', None, ['cval', 'dval', 'eval', 'fval'], 'oval', 'oval')

    def test_no_lcas(self):
        self.assertLCAMultiWay('this', 'bval', [], 'bval', 'tval')
        self.assertLCAMultiWay('other', 'bval', [], 'oval', 'bval')
        self.assertLCAMultiWay('conflict', 'bval', [], 'oval', 'tval')
        self.assertLCAMultiWay('this', 'bval', [], 'oval', 'oval')

    def test_lca_supersedes_other_lca(self):
        """If one lca == base, the other lca takes precedence"""
        self.assertLCAMultiWay('this', 'bval', ['bval', 'lcaval'], 'lcaval', 'tval')
        self.assertLCAMultiWay('this', 'bval', ['bval', 'lcaval'], 'lcaval', 'bval')
        self.assertLCAMultiWay('other', 'bval', ['bval', 'lcaval'], 'bval', 'lcaval')
        self.assertLCAMultiWay('conflict', 'bval', ['bval', 'lcaval'], 'bval', 'tval')

    def test_other_and_this_pick_different_lca(self):
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'lca1val', 'lca2val')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'lca1val', 'lca2val')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'bval'], 'lca1val', 'lca2val')

    def test_other_in_lca(self):
        self.assertLCAMultiWay('this', 'bval', ['lca1val', 'lca2val'], 'lca1val', 'newval')
        self.assertLCAMultiWay('this', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'lca1val', 'newval')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'lca1val', 'newval', allow_overriding_lca=False)
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'lca1val', 'newval', allow_overriding_lca=False)
        self.assertLCAMultiWay('this', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'lca1val', 'bval')
        self.assertLCAMultiWay('this', 'bval', ['lca1val', 'lca2val', 'bval'], 'lca1val', 'bval')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'lca1val', 'bval', allow_overriding_lca=False)
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'bval'], 'lca1val', 'bval', allow_overriding_lca=False)

    def test_this_in_lca(self):
        self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca1val')
        self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca2val')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca1val', allow_overriding_lca=False)
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'oval', 'lca2val', allow_overriding_lca=False)
        self.assertLCAMultiWay('other', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'bval', 'lca3val')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'bval', 'lca3val', allow_overriding_lca=False)

    def test_all_differ(self):
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val'], 'oval', 'tval')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca2val'], 'oval', 'tval')
        self.assertLCAMultiWay('conflict', 'bval', ['lca1val', 'lca2val', 'lca3val'], 'oval', 'tval')