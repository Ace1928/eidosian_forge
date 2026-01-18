import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalVolumeConnector(command.ShowOne):
    """Create a new baremetal volume connector."""
    log = logging.getLogger(__name__ + '.CreateBaremetalVolumeConnector')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalVolumeConnector, self).get_parser(prog_name)
        parser.add_argument('--node', dest='node_uuid', metavar='<uuid>', required=True, help=_('UUID of the node that this volume connector belongs to.'))
        parser.add_argument('--type', dest='type', metavar='<type>', required=True, choices=('iqn', 'ip', 'mac', 'wwnn', 'wwpn', 'port', 'portgroup'), help=_("Type of the volume connector. Can be 'iqn', 'ip', 'mac', 'wwnn', 'wwpn', 'port', 'portgroup'."))
        parser.add_argument('--connector-id', dest='connector_id', required=True, metavar='<connector id>', help=_("ID of the volume connector in the specified type. For example, the iSCSI initiator IQN for the node if the type is 'iqn'."))
        parser.add_argument('--uuid', dest='uuid', metavar='<uuid>', help=_('UUID of the volume connector.'))
        parser.add_argument('--extra', dest='extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)' % parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        field_list = ['extra', 'type', 'connector_id', 'node_uuid', 'uuid']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        fields = utils.args_array_to_dict(fields, 'extra')
        volume_connector = baremetal_client.volume_connector.create(**fields)
        data = dict([(f, getattr(volume_connector, f, '')) for f in res_fields.VOLUME_CONNECTOR_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)