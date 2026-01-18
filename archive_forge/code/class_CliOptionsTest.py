import collections
import json
import optparse
import sys
from unittest import mock
import testtools
from troveclient.compat import common
class CliOptionsTest(testtools.TestCase):

    def check_default_options(self, co):
        self.assertIsNone(co.username)
        self.assertIsNone(co.apikey)
        self.assertIsNone(co.tenant_id)
        self.assertIsNone(co.auth_url)
        self.assertEqual('keystone', co.auth_type)
        self.assertEqual('database', co.service_type)
        self.assertEqual('RegionOne', co.region)
        self.assertIsNone(co.service_url)
        self.assertFalse(co.insecure)
        self.assertFalse(co.verbose)
        self.assertFalse(co.debug)
        self.assertIsNone(co.token)

    def check_option(self, oparser, option_name):
        option = oparser.get_option('--%s' % option_name)
        self.assertIsNotNone(option)
        if option_name in common.CliOptions.DEFAULT_VALUES:
            self.assertEqual(common.CliOptions.DEFAULT_VALUES[option_name], option.default)

    def test___init__(self):
        co = common.CliOptions()
        self.check_default_options(co)

    def test_default(self):
        co = common.CliOptions.default()
        self.check_default_options(co)

    def test_load_from_file(self):
        co = common.CliOptions.load_from_file()
        self.check_default_options(co)

    def test_create_optparser(self):
        option_names = ['verbose', 'debug', 'auth_url', 'username', 'apikey', 'tenant_id', 'auth_type', 'service_type', 'service_name', 'service_type', 'service_name', 'service_url', 'region', 'insecure', 'token', 'secure', 'json', 'terse', 'hide-debug']
        oparser = common.CliOptions.create_optparser(True)
        for option_name in option_names:
            self.check_option(oparser, option_name)
        oparser = common.CliOptions.create_optparser(False)
        for option_name in option_names:
            self.check_option(oparser, option_name)