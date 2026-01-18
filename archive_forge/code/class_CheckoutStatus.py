import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
class CheckoutStatus(BranchStatus):

    def setUp(self):
        super().setUp()
        mkdir('codir')
        chdir('codir')

    def make_branch_and_tree(self, relpath):
        source = self.make_branch(pathjoin('..', relpath))
        checkout = bzrdir.BzrDirMetaFormat1().initialize(relpath)
        checkout.set_branch_reference(source)
        return checkout.create_workingtree()