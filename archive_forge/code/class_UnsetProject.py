import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class UnsetProject(command.Command):
    _description = _('Unset project properties')

    def get_parser(self, prog_name):
        parser = super(UnsetProject, self).get_parser(prog_name)
        parser.add_argument('project', metavar='<project>', help=_('Project to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', default=[], help=_('Unset a project property (repeat option to unset multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        project = utils.find_resource(identity_client.tenants, parsed_args.project)
        kwargs = project._info
        for key in parsed_args.property:
            if key in kwargs:
                kwargs[key] = None
        identity_client.tenants.update(project.id, **kwargs)