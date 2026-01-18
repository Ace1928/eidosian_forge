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
class ShowCommand(NeutronCommand, show.ShowOne):
    """Show information of a given resource."""
    log = None
    allow_names = True
    help_resource = None

    def get_parser(self, prog_name):
        parser = super(ShowCommand, self).get_parser(prog_name)
        add_show_list_common_argument(parser)
        if self.allow_names:
            help_str = _('ID or name of %s to look up.')
        else:
            help_str = _('ID of %s to look up.')
        if not self.help_resource:
            self.help_resource = self.resource
        parser.add_argument('id', metavar=self.resource.upper(), help=help_str % self.help_resource)
        self.add_known_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        self.set_extra_attrs(parsed_args)
        neutron_client = self.get_client()
        params = {}
        if parsed_args.show_details:
            params = {'verbose': 'True'}
        if parsed_args.fields:
            params = {'fields': parsed_args.fields}
        if self.allow_names:
            _id = find_resourceid_by_name_or_id(neutron_client, self.resource, parsed_args.id, cmd_resource=self.cmd_resource, parent_id=self.parent_id)
        else:
            _id = parsed_args.id
        obj_shower = getattr(neutron_client, 'show_%s' % self.cmd_resource)
        if self.parent_id:
            data = obj_shower(_id, self.parent_id, **params)
        else:
            data = obj_shower(_id, **params)
        self.cleanup_output_data(data)
        if parsed_args.formatter == 'table':
            self.format_output_data(data)
        resource = data[self.resource]
        if self.resource in data:
            return zip(*sorted(resource.items()))
        else:
            return None