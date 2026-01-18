import sys
import fire
from fire import testutils
import mock
class FireImportTest(testutils.BaseTestCase):
    """Tests importing Fire."""

    def testFire(self):
        with mock.patch.object(sys, 'argv', ['commandname']):
            fire.Fire()

    def testFireMethods(self):
        self.assertIsNotNone(fire.Fire)

    def testNoPrivateMethods(self):
        self.assertTrue(hasattr(fire, 'Fire'))
        self.assertFalse(hasattr(fire, '_Fire'))