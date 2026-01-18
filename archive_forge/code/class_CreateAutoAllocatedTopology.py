import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class CreateAutoAllocatedTopology(command.ShowOne):
    _description = _('Create the  auto allocated topology for project')

    def get_parser(self, prog_name):
        parser = super(CreateAutoAllocatedTopology, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_('Return the auto allocated topology for a given project. Default is current project'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--check-resources', action='store_true', help=_('Validate the requirements for auto allocated topology. Does not return a topology.'))
        parser.add_argument('--or-show', action='store_true', default=True, help=_("If topology exists returns the topology's information (Default)"))
        return parser

    def check_resource_topology(self, client, parsed_args):
        obj = client.validate_auto_allocated_topology(parsed_args.project)
        columns = _format_check_resource_columns()
        data = utils.get_item_properties(_format_check_resource(obj), columns, formatters={})
        return (columns, data)

    def get_topology(self, client, parsed_args):
        obj = client.get_auto_allocated_topology(parsed_args.project)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        if parsed_args.check_resources:
            columns, data = self.check_resource_topology(client, parsed_args)
        else:
            columns, data = self.get_topology(client, parsed_args)
        return (columns, data)