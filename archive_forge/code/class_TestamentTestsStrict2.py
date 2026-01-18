import os
from breezy import osutils
from breezy.bzr.testament import StrictTestament, StrictTestament3, Testament
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import SymlinkFeature
class TestamentTestsStrict2(TestamentTests):

    def testament_class(self):
        return StrictTestament3