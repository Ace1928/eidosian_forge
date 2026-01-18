import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class AddProjectToEndpointGroup(command.Command):
    _description = _('Add a project to an endpoint group')

    def get_parser(self, prog_name):
        parser = super(AddProjectToEndpointGroup, self).get_parser(prog_name)
        parser.add_argument('endpointgroup', metavar='<endpoint-group>', help=_('Endpoint group (name or ID)'))
        parser.add_argument('project', metavar='<project>', help=_('Project to associate (name or ID)'))
        common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.identity
        endpointgroup = utils.find_resource(client.endpoint_groups, parsed_args.endpointgroup)
        project = common.find_project(client, parsed_args.project, parsed_args.project_domain)
        client.endpoint_filter.add_endpoint_group_to_project(endpoint_group=endpointgroup.id, project=project.id)