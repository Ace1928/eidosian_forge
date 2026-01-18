import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class UnsetQos(command.Command):
    _description = _('Unset QoS specification properties')

    def get_parser(self, prog_name):
        parser = super(UnsetQos, self).get_parser(prog_name)
        parser.add_argument('qos_spec', metavar='<qos-spec>', help=_('QoS specification to modify (name or ID)'))
        parser.add_argument('--property', metavar='<key>', action='append', default=[], help=_('Property to remove from the QoS specification. (repeat option to unset multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_spec = utils.find_resource(volume_client.qos_specs, parsed_args.qos_spec)
        if parsed_args.property:
            volume_client.qos_specs.unset_keys(qos_spec.id, parsed_args.property)