from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertStringToCaseInsensitive(base.BaseTestCase):

    def test_convert_string_to_lower(self):
        result = converters.convert_string_to_case_insensitive(u'THIS Is tEsT')
        self.assertIsInstance(result, str)

    def test_assert_error_on_non_string(self):
        for invalid in [[], 123]:
            with testtools.ExpectedException(n_exc.InvalidInput):
                converters.convert_string_to_case_insensitive(invalid)