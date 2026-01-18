from unittest import mock
import novaclient
from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient import utils as nutils
from novaclient.v2 import versions
class DecoratedAfterTestCase(utils.TestCase):

    def test_decorated_after(self):

        class Fake(object):
            api_version = api_versions.APIVersion('2.123')

            @api_versions.deprecated_after('2.123')
            def foo(self):
                pass
        with mock.patch('warnings.warn') as mock_warn:
            Fake().foo()
            msg = 'The novaclient.tests.unit.test_api_versions module is deprecated and will be removed.'
            mock_warn.assert_called_once_with(msg, mock.ANY)