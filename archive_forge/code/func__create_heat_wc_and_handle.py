import datetime
from unittest import mock
import uuid
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from heat.common import identifier
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import swift as swift_plugin
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.openstack.heat import wait_condition_handle as h_wch
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.objects import resource as resource_objects
from heat.tests import common
from heat.tests import utils
def _create_heat_wc_and_handle(self):
    self.stack = self.create_stack(template=test_template_heat_waitcondition)
    mock_get_status = h_wch.HeatWaitConditionHandle.get_status
    mock_get_status.side_effect = [['SUCCESS']]
    self.stack.create()
    rsrc = self.stack['wait_condition']
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    wc_att = rsrc.FnGetAtt('data')
    self.assertEqual(str({}), wc_att)
    handle = self.stack['wait_handle']
    self.assertEqual((handle.CREATE, handle.COMPLETE), handle.state)
    return (rsrc, handle, mock_get_status)