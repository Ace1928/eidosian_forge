import abc
import argparse
import functools
import logging
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
class NeutronCommand(command.Command, metaclass=NeutronCommandMeta):
    values_specs = []
    json_indent = None
    resource = None
    shadow_resource = None
    parent_id = None

    def run(self, parsed_args):
        self.log.debug('run(%s)', parsed_args)
        return super(NeutronCommand, self).run(parsed_args)

    @property
    def cmd_resource(self):
        if self.shadow_resource:
            return self.shadow_resource
        return self.resource

    def get_client(self):
        return self.app.client_manager.neutron

    def get_parser(self, prog_name):
        parser = super(NeutronCommand, self).get_parser(prog_name)
        parser.add_argument('--request-format', help=argparse.SUPPRESS, default='json', choices=['json'])
        parser.add_argument('--request_format', choices=['json'], help=argparse.SUPPRESS)
        return parser

    def cleanup_output_data(self, data):
        pass

    def format_output_data(self, data):
        if self.resource in data:
            for k, v in data[self.resource].items():
                if isinstance(v, list):
                    value = '\n'.join((jsonutils.dumps(i, indent=self.json_indent) if isinstance(i, dict) else str(i) for i in v))
                    data[self.resource][k] = value
                elif isinstance(v, dict):
                    value = jsonutils.dumps(v, indent=self.json_indent)
                    data[self.resource][k] = value
                elif v is None:
                    data[self.resource][k] = ''

    def add_known_arguments(self, parser):
        pass

    def set_extra_attrs(self, parsed_args):
        pass

    def args2body(self, parsed_args):
        return {}