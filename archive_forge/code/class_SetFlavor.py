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
class SetFlavor(command.Command):
    _description = _('Set flavor properties')

    def get_parser(self, prog_name):
        parser = super(SetFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', help=_('Flavor to modify (name or ID)'))
        parser.add_argument('--no-property', action='store_true', help=_('Remove all properties from this flavor (specify both --no-property and --property to remove the current properties before setting new properties.)'))
        parser.add_argument('--property', metavar='<key=value>', action=parseractions.KeyValueAction, dest='properties', help=_('Property to add or modify for this flavor (repeat option to set multiple properties)'))
        parser.add_argument('--project', metavar='<project>', help=_('Set flavor access to project (name or ID) (admin only)'))
        identity_common.add_project_domain_option_to_parser(parser)
        parser.add_argument('--description', metavar='<description>', help=_("Set description for the flavor.(Supported by API versions '2.55' - '2.latest'"))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        identity_client = self.app.client_manager.identity
        try:
            flavor = compute_client.find_flavor(parsed_args.flavor, get_extra_specs=True, ignore_missing=False)
        except sdk_exceptions.ResourceNotFound as e:
            raise exceptions.CommandError(e.message)
        if parsed_args.description:
            if not sdk_utils.supports_microversion(compute_client, '2.55'):
                msg = _('The --description parameter requires server support for API microversion 2.55')
                raise exceptions.CommandError(msg)
            compute_client.update_flavor(flavor=flavor.id, description=parsed_args.description)
        result = 0
        if parsed_args.no_property:
            try:
                for key in flavor.extra_specs.keys():
                    compute_client.delete_flavor_extra_specs_property(flavor.id, key)
            except Exception as e:
                LOG.error(_('Failed to clear flavor properties: %s'), e)
                result += 1
        if parsed_args.properties:
            try:
                compute_client.create_flavor_extra_specs(flavor.id, parsed_args.properties)
            except Exception as e:
                LOG.error(_('Failed to set flavor properties: %s'), e)
                result += 1
        if parsed_args.project:
            try:
                if flavor.is_public:
                    msg = _('Cannot set access for a public flavor')
                    raise exceptions.CommandError(msg)
                else:
                    project_id = identity_common.find_project(identity_client, parsed_args.project, parsed_args.project_domain).id
                    compute_client.flavor_add_tenant_access(flavor.id, project_id)
            except Exception as e:
                LOG.error(_('Failed to set flavor access to project: %s'), e)
                result += 1
        if result > 0:
            raise exceptions.CommandError(_('Command Failed: One or more of the operations failed'))