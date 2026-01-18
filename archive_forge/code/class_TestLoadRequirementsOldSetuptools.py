import importlib.metadata as importlib_metadata
import operator
from unittest import mock
from stevedore import exception
from stevedore import extension
from stevedore.tests import utils
class TestLoadRequirementsOldSetuptools(utils.TestCase):

    def setUp(self):
        super(TestLoadRequirementsOldSetuptools, self).setUp()
        self.mock_ep = mock.Mock(spec=['load', 'name'])
        self.em = extension.ExtensionManager.make_test_instance([])

    def test_verify_requirements(self):
        self.em._load_one_plugin(self.mock_ep, False, (), {}, verify_requirements=True)
        self.mock_ep.load.assert_called_once_with()

    def test_no_verify_requirements(self):
        self.em._load_one_plugin(self.mock_ep, False, (), {}, verify_requirements=False)
        self.mock_ep.load.assert_called_once_with()