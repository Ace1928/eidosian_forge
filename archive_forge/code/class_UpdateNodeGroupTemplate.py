import sys
from osc_lib import utils as osc_utils
from saharaclient.osc import utils
from saharaclient.osc.v1 import node_group_templates as ngt_v1
class UpdateNodeGroupTemplate(ngt_v1.UpdateNodeGroupTemplate, utils.NodeGroupTemplatesUtils):
    """Updates node group template"""

    def get_parser(self, prog_name):
        parser = super(UpdateNodeGroupTemplate, self).get_parser(prog_name)
        bootfromvolume = parser.add_mutually_exclusive_group()
        bootfromvolume.add_argument('--boot-from-volume-enable', action='store_true', help='Makes node group bootable from volume.', dest='boot_from_volume')
        bootfromvolume.add_argument('--boot-from-volume-disable', action='store_false', help='Makes node group not bootable from volume.', dest='boot_from_volume')
        parser.add_argument('--boot-volume-type', metavar='<boot-volume-type>', help='Type of the boot volume. This parameter will be taken into account only if booting from volume.')
        parser.add_argument('--boot-volume-availability-zone', metavar='<boot-volume-availability-zone>', help='Name of the availability zone to create boot volume in. This parameter will be taken into account only if booting from volume.')
        bfv_locality = parser.add_mutually_exclusive_group()
        bfv_locality.add_argument('--boot-volume-local-to-instance-enable', action='store_true', help='Makes boot volume explicitly local to instance.', dest='boot_volume_local_to_instance')
        bfv_locality.add_argument('--boot-volume-local-to-instance-disable', action='store_false', help='Removes explicit instruction of boot volume locality.', dest='boot_volume_local_to_instance')
        parser.set_defaults(is_public=None, is_protected=None, is_proxy_gateway=None, volume_locality=None, use_auto_security_group=None, use_autoconfig=None, boot_from_volume=None, boot_volume_type=None, boot_volume_availability_zone=None, boot_volume_local_to_instance=None)
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.data_processing
        data = self._update_take_action(client, self.app, parsed_args)
        _format_ngt_output(data)
        data = utils.prepare_data(data, NGT_FIELDS)
        return self.dict2columns(data)