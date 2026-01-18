import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
class RootHistoryTest(testtools.TestCase):

    def setUp(self):
        super(RootHistoryTest, self).setUp()
        self.orig__init = management.RootHistory.__init__
        management.RootHistory.__init__ = mock.Mock(return_value=None)

    def tearDown(self):
        super(RootHistoryTest, self).tearDown()
        management.RootHistory.__init__ = self.orig__init

    def test___repr__(self):
        root_history = management.RootHistory()
        root_history.id = '1'
        root_history.created = 'ct'
        root_history.user = 'tu'
        self.assertEqual('<Root History: Instance 1 enabled at ct by tu>', root_history.__repr__())