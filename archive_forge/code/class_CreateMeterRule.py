import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class CreateMeterRule(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create a new meter rule')

    def get_parser(self, prog_name):
        parser = super(CreateMeterRule, self).get_parser(prog_name)
        parser.add_argument('--project', metavar='<project>', help=_("Owner's project (name or ID)"))
        identity_common.add_project_domain_option_to_parser(parser)
        exclude_group = parser.add_mutually_exclusive_group()
        exclude_group.add_argument('--exclude', action='store_true', help=_('Exclude remote IP prefix from traffic count'))
        exclude_group.add_argument('--include', action='store_true', help=_('Include remote IP prefix from traffic count (default)'))
        direction_group = parser.add_mutually_exclusive_group()
        direction_group.add_argument('--ingress', action='store_true', help=_('Apply rule to incoming network traffic (default)'))
        direction_group.add_argument('--egress', action='store_true', help=_('Apply rule to outgoing network traffic'))
        parser.add_argument('--remote-ip-prefix', metavar='<remote-ip-prefix>', required=False, help=_('The remote IP prefix to associate with this rule'))
        parser.add_argument('--source-ip-prefix', metavar='<remote-ip-prefix>', required=False, help=_('The source IP prefix to associate with this rule'))
        parser.add_argument('--destination-ip-prefix', metavar='<remote-ip-prefix>', required=False, help=_('The destination IP prefix to associate with this rule'))
        parser.add_argument('meter', metavar='<meter>', help=_('Label to associate with this metering rule (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        _meter = client.find_metering_label(parsed_args.meter, ignore_missing=False)
        parsed_args.meter = _meter.id
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        obj = client.create_metering_label_rule(**attrs)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns, formatters={})
        return (display_columns, data)