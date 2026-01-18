import os
import shutil
from io import StringIO
from .. import bisect
from ..controldir import ControlDir
from . import TestCaseWithTransport, TestSkipped
class BisectCurrentUnitTests(BisectTestCase):
    """Test the BisectCurrent class."""

    def testShowLog(self):
        """Test that the log can be shown."""
        sio = StringIO()
        bisect.BisectCurrent(self.tree.controldir).show_rev_log(outf=sio)

    def testShowLogSubtree(self):
        """Test that a subtree's log can be shown."""
        current = bisect.BisectCurrent(self.tree.controldir)
        current.switch(self.subtree_rev)
        sio = StringIO()
        current.show_rev_log(outf=sio)

    def testSwitchVersions(self):
        """Test switching versions."""
        current = bisect.BisectCurrent(self.tree.controldir)
        self.assertRevno(5)
        current.switch(4)
        self.assertRevno(4)

    def testReset(self):
        """Test resetting the working tree to a non-bisected state."""
        current = bisect.BisectCurrent(self.tree.controldir)
        current.switch(4)
        current.reset()
        self.assertRevno(5)
        self.assertFalse(os.path.exists(os.path.join('.bzr', bisect.BISECT_REV_PATH)))

    def testIsMergePoint(self):
        """Test merge point detection."""
        current = bisect.BisectCurrent(self.tree.controldir)
        self.assertRevno(5)
        self.assertFalse(current.is_merge_point())
        current.switch(2)
        self.assertTrue(current.is_merge_point())