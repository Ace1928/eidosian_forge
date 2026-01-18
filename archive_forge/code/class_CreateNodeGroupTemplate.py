import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class CreateNodeGroupTemplate(ngt_v1.CreateNodeGroupTemplate, utils.NodeGroupTemplatesUtils):
    """Creates node group template"""

    def get_parser(self, prog_name):
        parser = super(CreateNodeGroupTemplate, self).get_parser(prog_name)
        parser.add_argument('--boot-from-volume', action='store_true', default=False, help='Make the node group bootable from volume')
        parser.add_argument('--boot-volume-type', metavar='<boot-volume-type>', help='Type of the boot volume. This parameter will be taken into account only if booting from volume.')
        parser.add_argument('--boot-volume-availability-zone', metavar='<boot-volume-availability-zone>', help='Name of the availability zone to create boot volume in. This parameter will be taken into account only if booting from volume.')
        parser.add_argument('--boot-volume-local-to-instance', action='store_true', default=False, help='Instance and volume guaranteed on the same host. This parameter will be taken into account only if booting from volume.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = self._create_take_action(client, self.app, parsed_args)
        _format_ngt_output(data)
        data = utils.prepare_data(data, NGT_FIELDS)
        return self.dict2columns(data)