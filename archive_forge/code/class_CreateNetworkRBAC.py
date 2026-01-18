import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateNetworkRBAC(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create network RBAC policy')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkRBAC, self).get_parser(prog_name)
        parser.add_argument('rbac_object', metavar='<rbac-object>', help=_('The object to which this RBAC policy affects (name or ID)'))
        parser.add_argument('--type', metavar='<type>', required=True, choices=['address_group', 'address_scope', 'security_group', 'subnetpool', 'qos_policy', 'network'], help=_('Type of the object that RBAC policy affects ("address_group", "address_scope", "security_group", "subnetpool", "qos_policy" or "network")'))
        parser.add_argument('--action', metavar='<action>', required=True, choices=['access_as_external', 'access_as_shared'], help=_('Action for the RBAC policy ("access_as_external" or "access_as_shared")'))
        target_project_group = parser.add_mutually_exclusive_group(required=True)
        target_project_group.add_argument('--target-project', metavar='<target-project>', help=_('The project to which the RBAC policy will be enforced (name or ID)'))
        target_project_group.add_argument('--target-all-projects', action='store_true', help=_('Allow creating RBAC policy for all projects.'))
        parser.add_argument('--target-project-domain', metavar='<target-project-domain>', help=_('Domain the target project belongs to (name or ID). This can be used in case collisions between project names exist.'))
        parser.add_argument('--project', metavar='<project>', help=_('The owner project (name or ID)'))
        identity_common.add_project_domain_option_to_parser(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_rbac_policy(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)