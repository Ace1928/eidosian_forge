import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateProject(command.ShowOne):
    _description = _('Create new project')

    def get_parser(self, prog_name):
        parser = super(CreateProject, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<project-name>', help=_('New project name'))
        parser.add_argument('--description', metavar='<description>', help=_('Project description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--enable', action='store_true', help=_('Enable project (default)'))
        enable_group.add_argument('--disable', action='store_true', help=_('Disable project'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, help=_('Add a property to <name> (repeat option to set multiple properties)'))
        parser.add_argument('--or-show', action='store_true', help=_('Return existing project'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        enabled = True
        if parsed_args.disable:
            enabled = False
        kwargs = {}
        if parsed_args.property:
            kwargs = parsed_args.property.copy()
        try:
            project = identity_client.tenants.create(parsed_args.name, description=parsed_args.description, enabled=enabled, **kwargs)
        except ks_exc.Conflict:
            if parsed_args.or_show:
                project = utils.find_resource(identity_client.tenants, parsed_args.name)
                LOG.info(_('Returning existing project %s'), project.name)
            else:
                raise
        project._info.pop('parent_id', None)
        return zip(*sorted(project._info.items()))