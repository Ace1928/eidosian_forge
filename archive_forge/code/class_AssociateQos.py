import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class AssociateQos(command.Command):
    _description = _('Associate a QoS specification to a volume type')

    def get_parser(self, prog_name):
        parser = super(AssociateQos, self).get_parser(prog_name)
        parser.add_argument('qos_spec', metavar='<qos-spec>', help=_('QoS specification to modify (name or ID)'))
        parser.add_argument('volume_type', metavar='<volume-type>', help=_('Volume type to associate the QoS (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        qos_spec = utils.find_resource(volume_client.qos_specs, parsed_args.qos_spec)
        volume_type = utils.find_resource(volume_client.volume_types, parsed_args.volume_type)
        volume_client.qos_specs.associate(qos_spec.id, volume_type.id)