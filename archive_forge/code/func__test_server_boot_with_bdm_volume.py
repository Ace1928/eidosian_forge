import itertools
import json
import time
import uuid
from tempest.lib import exceptions
from openstackclient.compute.v2 import server as v2_server
from openstackclient.tests.functional.compute.v2 import common
from openstackclient.tests.functional.volume.v2 import common as volume_common
def _test_server_boot_with_bdm_volume(self, use_legacy):
    """Test server create from volume, server delete"""
    volume_wait_for = volume_common.BaseVolumeTests.wait_for_status
    volume_name = uuid.uuid4().hex
    cmd_output = self.openstack('volume create ' + '--size 1 ' + volume_name, parse_output=True)
    volume_id = cmd_output['id']
    self.assertIsNotNone(volume_id)
    self.addCleanup(self.openstack, 'volume delete ' + volume_name)
    self.assertEqual(volume_name, cmd_output['name'])
    volume_wait_for('volume', volume_name, 'available')
    if use_legacy:
        bdm_arg = f'--block-device-mapping vdb={volume_name}'
    else:
        bdm_arg = f'--block-device device_name=vdb,source_type=volume,boot_index=1,uuid={volume_id}'
    server_name = uuid.uuid4().hex
    server = self.openstack('server create ' + '--flavor ' + self.flavor_name + ' ' + '--image ' + self.image_name + ' ' + bdm_arg + ' ' + self.network_arg + ' ' + '--wait ' + server_name, parse_output=True)
    self.assertIsNotNone(server['id'])
    self.addCleanup(self.openstack, 'server delete --wait ' + server_name)
    self.assertEqual(server_name, server['name'])
    cmd_output = self.openstack('server show ' + server_name, parse_output=True)
    volumes_attached = cmd_output['volumes_attached']
    self.assertIsNotNone(volumes_attached)
    cmd_output = self.openstack('volume show ' + volume_name, parse_output=True)
    attachments = cmd_output['attachments']
    self.assertEqual(1, len(attachments))
    self.assertEqual(server['id'], attachments[0]['server_id'])
    self.assertEqual('in-use', cmd_output['status'])