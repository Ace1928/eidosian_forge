import copy
from unittest import mock
from ironicclient.common.apiclient import exceptions as ic_exc
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import ironic as ic
from heat.engine import resource
from heat.engine.resources.openstack.ironic import port
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _mock_get_client(self):
    value = mock.MagicMock(address=self.fake_address, node_uuid=self.fake_node_uuid, portgroup_uuid=self.fake_portgroup_uuid, local_link_connection=self.fake_local_link_connection, pxe_enabled=self.fake_pxe_enabled, physical_network=self.fake_physical_network, internal_info=self.fake_internal_info, extra=self.fake_extra, is_smartnic=self.fake_is_smartnic, uuid=self.resource_id)
    value.to_dict.return_value = value.__dict__
    self.client.port.get.return_value = value