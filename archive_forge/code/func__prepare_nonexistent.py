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
def _prepare_nonexistent(self):
    wt = self.make_branch_and_tree('.')
    self.assertStatus([], wt)
    self.build_tree(['FILE_A', 'FILE_B', 'FILE_C', 'FILE_D', 'FILE_E'])
    wt.add('FILE_A')
    wt.add('FILE_B')
    wt.add('FILE_C')
    wt.add('FILE_D')
    wt.add('FILE_E')
    wt.commit('Create five empty files.')
    with open('FILE_B', 'w') as f:
        f.write('Modification to file FILE_B.')
    with open('FILE_C', 'w') as f:
        f.write('Modification to file FILE_C.')
    unlink('FILE_E')
    with open('FILE_Q', 'w') as f:
        f.write('FILE_Q is added but not committed.')
    wt.add('FILE_Q')
    open('UNVERSIONED_BUT_EXISTING', 'w')
    return wt