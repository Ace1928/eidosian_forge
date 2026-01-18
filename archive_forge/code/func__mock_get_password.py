import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
@staticmethod
@contextlib.contextmanager
def _mock_get_password(*passwords):
    mocker = mock.Mock(side_effect=passwords)
    with mock.patch('osc_lib.utils.get_password', mocker):
        yield