import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
@ddt.ddt
class AllowOnlyOneAliasAtATimeActionTest(utils.TestCase):
    FAKE_ENV = {'OS_USERNAME': 'username', 'OS_PASSWORD': 'password', 'OS_TENANT_NAME': 'tenant_name', 'OS_AUTH_URL': 'http://no.where'}

    def setUp(self):
        super(self.__class__, self).setUp()
        for k, v in self.FAKE_ENV.items():
            self.useFixture(fixtures.EnvironmentVariable(k, v))
        self.mock_object(shell.client, 'get_client_class', mock.Mock(return_value=fakes.FakeClient))

    def shell_discover_client(self, current_client, os_api_version, os_endpoint_type, os_service_type, client_args):
        return (current_client, manilaclient.API_MAX_VERSION)

    def shell(self, argstr):
        orig = sys.stdout
        try:
            sys.stdout = io.StringIO()
            _shell = CustomOpenStackManilaShell()
            _shell._discover_client = self.shell_discover_client
            _shell.main(argstr.split())
        except SystemExit:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self.assertEqual(exc_value.code, 0)
        finally:
            out = sys.stdout.getvalue()
            sys.stdout.close()
            sys.stdout = orig
        return out

    @ddt.data(('--default-is-none foo', 'foo'), ('--default-is-none foo --default-is-none foo', 'foo'), ('--default-is-none foo --default_is_none foo', 'foo'), ('--default_is_none None', 'None'))
    @ddt.unpack
    def test_foo_success(self, options_str, expected_result):
        output = self.shell('foo %s' % options_str)
        parsed_output = output_parser.details(output)
        self.assertEqual({'key': expected_result}, parsed_output)

    @ddt.data('--default-is-none foo --default-is-none bar', '--default-is-none foo --default_is_none bar', '--default-is-none foo --default_is_none FOO')
    def test_foo_error(self, options_str):
        self.assertRaises(matchers.MismatchError, self.shell, 'foo %s' % options_str)

    @ddt.data(('--default-is-not-none bar', 'bar'), ('--default_is_not_none bar --default-is-not-none bar', 'bar'), ('--default_is_not_none bar --default_is_not_none bar', 'bar'), ('--default-is-not-none not_bar', 'not_bar'), ('--default_is_not_none None', 'None'))
    @ddt.unpack
    def test_bar_success(self, options_str, expected_result):
        output = self.shell('bar %s' % options_str)
        parsed_output = output_parser.details(output)
        self.assertEqual({'key': expected_result}, parsed_output)

    @ddt.data('--default-is-not-none foo --default-is-not-none bar', '--default-is-not-none foo --default_is_not_none bar', '--default-is-not-none bar --default_is_not_none BAR')
    def test_bar_error(self, options_str):
        self.assertRaises(matchers.MismatchError, self.shell, 'bar %s' % options_str)

    @ddt.data(('--list-like q=w', "['q=w']"), ('--list-like q=w --list_like q=w', "['q=w']"), ('--list-like q=w e=r t=y --list_like e=r t=y q=w', "['e=r', 'q=w', 't=y']"), ('--list_like q=w e=r t=y', "['e=r', 'q=w', 't=y']"))
    @ddt.unpack
    def test_quuz_success(self, options_str, expected_result):
        output = self.shell('quuz %s' % options_str)
        parsed_output = output_parser.details(output)
        self.assertEqual({'key': expected_result}, parsed_output)

    @ddt.data('--list-like q=w --list_like e=r t=y')
    def test_quuz_error(self, options_str):
        self.assertRaises(matchers.MismatchError, self.shell, 'quuz %s' % options_str)