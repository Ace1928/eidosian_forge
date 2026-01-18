import random
import string
from tempest.lib import decorators
from novaclient.tests.functional import base
from novaclient.tests.functional.v2.legacy import test_servers
from novaclient.v2 import shell
class TestServeRebuildV275(TestServeRebuildV274):
    COMPUTE_API_VERSION = '2.75'
    REBUILD_FIELDS_V275 = ['OS-EXT-AZ:availability_zone', 'config_drive', 'OS-EXT-SRV-ATTR:host', 'OS-EXT-SRV-ATTR:hypervisor_hostname', 'OS-EXT-SRV-ATTR:instance_name', 'OS-EXT-SRV-ATTR:hostname', 'OS-EXT-SRV-ATTR:kernel_id', 'OS-EXT-SRV-ATTR:launch_index', 'OS-EXT-SRV-ATTR:ramdisk_id', 'OS-EXT-SRV-ATTR:reservation_id', 'OS-EXT-SRV-ATTR:root_device_name', 'host_status', 'OS-SRV-USG:launched_at', 'OS-SRV-USG:terminated_at', 'OS-EXT-STS:task_state', 'OS-EXT-STS:vm_state', 'OS-EXT-STS:power_state', 'security_groups', 'os-extended-volumes:volumes_attached']
    REBUILD_FIELDS = TestServeRebuildV274.REBUILD_FIELDS + REBUILD_FIELDS_V275