from unittest import mock
from designateclient import exceptions as designate_exception
from heat.common import exception
from heat.engine.resources.openstack.designate import recordset
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _mock_check_status_active(self):
    self.test_client.recordsets.get.side_effect = [{'status': 'PENDING'}, {'status': 'ACTIVE'}, {'status': 'ERROR'}]