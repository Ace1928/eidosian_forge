import logging
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class UnsetFlavor(command.Command):
    _description = _('Unset flavor properties')

    def get_parser(self, prog_name):
        parser = super(UnsetFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', help=_('Flavor to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', dest='properties', help=_('Property to remove from flavor (repeat option to unset multiple properties)'))
        parser.add_argument('--project', metavar='<project>', help=_('Remove flavor access from project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        try:
            flavor = compute_client.find_flavor(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        except sdk_exceptions.ResourceNotFound as e:
            raise exceptions.CommandError(_(e.message))
        result = 0
        if parsed_args.properties:
            for key in parsed_args.properties:
                try:
                    compute_client.delete_flavor_extra_specs_property(flavor.id, key)
                except sdk_exceptions.SDKException as e:
                    LOG.error(_('Failed to unset flavor property: %s'), e)
                    result += 1
        if parsed_args.project:
            try:
                if flavor.is_public:
                    msg = _('Cannot remove access for a public flavor')
                    raise exceptions.CommandError(msg)
                project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
                compute_client.flavor_remove_tenant_access(flavor.id, project_id)
            except Exception as e:
                LOG.error(_('Failed to remove flavor access from project: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('Command Failed: One or more of the operations failed'))