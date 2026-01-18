import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateNetworkQosPolicy(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create a QoS policy')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkQosPolicy, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Name of QoS policy to create'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of the QoS policy'))
        share_group = parser.add_mutually_exclusive_group()
        share_group.add_argument('--share', action='store_true', default=None, help=_('Make the QoS policy accessible by other projects'))
        share_group.add_argument('--no-share', action='store_true', help=_('Make the QoS policy not accessible by other projects (default)'))
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        default_group = parser.add_mutually_exclusive_group()
        default_group.add_argument('--default', action='store_true', help=_('Set this as a default network QoS policy'))
        default_group.add_argument('--no-default', action='store_true', help=_('Set this as a non-default network QoS policy'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_qos_policy(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)