import os
import re
import unittest
from breezy import bzr, config, controldir, errors, osutils, repository, tests
from breezy.bzr.groupcompress_repo import RepositoryFormat2a
class TestOptParseBugHandling(tests.TestCase):
    """Test that we handle http://bugs.python.org/issue2931"""

    def test_nonascii_optparse(self):
        """Reasonable error raised when non-ascii in option name on Python 2"""
        error_re = 'no such option: -ä'
        out = self.run_bzr_error([error_re], ['st', '-ä'])