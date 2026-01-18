from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
def _create_connection_properties(self):
    return {'volume_id': 'ed083474-d325-4a99-b301-269111654f0d', 'volume': 'ref-1', 'vmdk_path': '[ds] foo/bar.vmdk', 'vmdk_size': units.Gi, 'datastore': 'ds-1', 'datacenter': 'dc-1'}