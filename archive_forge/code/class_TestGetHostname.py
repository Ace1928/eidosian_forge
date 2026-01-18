import itertools
import random
import socket
from unittest import mock
from neutron_lib import constants
from neutron_lib.tests import _base as base
from neutron_lib.utils import net
class TestGetHostname(base.BaseTestCase):

    @mock.patch.object(socket, 'gethostname', return_value='fake-host-name')
    def test_get_hostname(self, mock_gethostname):
        self.assertEqual('fake-host-name', net.get_hostname())
        mock_gethostname.assert_called_once_with()