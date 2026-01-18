from osc_lib.tests import utils
from unittest import mock
class TestDataProcessing(utils.TestCommand):

    def setUp(self):
        super(TestDataProcessing, self).setUp()
        self.app.client_manager.data_processing = mock.Mock()
        self.app.client_manager.network = mock.Mock()
        self.app.client_manager.compute = mock.Mock()