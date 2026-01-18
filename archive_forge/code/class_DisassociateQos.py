import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DisassociateQos(command.Command):
    _description = _('Disassociate a QoS specification from a volume type')

    def get_parser(self, prog_name):
        parser = super(DisassociateQos, self).get_parser(prog_name)
        parser.add_argument('qos_spec', metavar='<qos-spec>', help=_('QoS specification to modify (name or ID)'))
        volume_type_group = parser.add_mutually_exclusive_group()
        volume_type_group.add_argument('--volume-type', metavar='<volume-type>', help=_('Volume type to disassociate the QoS from (name or ID)'))
        volume_type_group.add_argument('--all', action='store_true', default=False, help=_('Disassociate the QoS from every volume type'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_spec = utils.find_resource(volume_client.qos_specs, parsed_args.qos_spec)
        if parsed_args.volume_type:
            volume_type = utils.find_resource(volume_client.volume_types, parsed_args.volume_type)
            volume_client.qos_specs.disassociate(qos_spec.id, volume_type.id)
        elif parsed_args.all:
            volume_client.qos_specs.disassociate_all(qos_spec.id)