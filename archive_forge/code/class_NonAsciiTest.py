import os
import sys
from unicodedata import normalize
from .. import osutils
from ..osutils import pathjoin
from . import TestCase, TestCaseWithTransport, TestSkipped
class NonAsciiTest(TestCaseWithTransport):

    def test_add_in_nonascii_branch(self):
        """Test adding in a non-ASCII branch."""
        br_dir = 'ሴ'
        try:
            wt = self.make_branch_and_tree(br_dir)
        except UnicodeEncodeError:
            raise TestSkipped("filesystem can't accomodate nonascii names")
            return
        with open(pathjoin(br_dir, 'a'), 'w') as f:
            f.write('hello')
        wt.add(['a'], ids=[b'a-id'])