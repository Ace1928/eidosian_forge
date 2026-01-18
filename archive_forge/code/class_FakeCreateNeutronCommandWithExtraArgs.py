import argparse
from unittest import mock
import openstack
from osc_lib import exceptions
from openstackclient.network import common
from openstackclient.tests.unit import utils
class FakeCreateNeutronCommandWithExtraArgs(common.NeutronCommandWithExtraArgs):

    def get_parser(self, prog_name):
        parser = super(FakeCreateNeutronCommandWithExtraArgs, self).get_parser(prog_name)
        parser.add_argument('--known-attribute')
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        if 'known_attribute' in parsed_args:
            attrs['known_attribute'] = parsed_args.known_attribute
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.test_create_action(**attrs)