import collections
import copy
import getpass
import json
import tempfile
from unittest import mock
from unittest.mock import call
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class _TestServerList(TestServer):
    columns = ('ID', 'Name', 'Status', 'Networks', 'Image', 'Flavor')
    columns_long = ('ID', 'Name', 'Status', 'Task State', 'Power State', 'Networks', 'Image Name', 'Image ID', 'Flavor Name', 'Flavor ID', 'Availability Zone', 'Host', 'Properties')

    def setUp(self):
        super(_TestServerList, self).setUp()
        self.kwargs = {'reservation_id': None, 'ip': None, 'ip6': None, 'name': None, 'status': None, 'flavor': None, 'image': None, 'host': None, 'project_id': None, 'all_projects': False, 'user_id': None, 'deleted': False, 'changes-since': None, 'changes-before': None}
        self.attrs = {'status': 'ACTIVE', 'OS-EXT-STS:task_state': 'None', 'OS-EXT-STS:power_state': 1, 'networks': {u'public': [u'10.20.30.40', u'2001:db8::5']}, 'OS-EXT-AZ:availability_zone': 'availability-zone-xxx', 'OS-EXT-SRV-ATTR:host': 'host-name-xxx', 'Metadata': format_columns.DictColumn({})}
        self.image = image_fakes.create_one_image()
        self.image_client.find_image.return_value = self.image
        self.image_client.get_image.return_value = self.image
        self.flavor = compute_fakes.create_one_flavor()
        self.compute_sdk_client.find_flavor.return_value = self.flavor
        self.attrs['flavor'] = {'original_name': self.flavor.name}
        self.servers = self.setup_sdk_servers_mock(3)
        self.compute_sdk_client.servers.return_value = self.servers
        self.cmd = server.ListServer(self.app, None)