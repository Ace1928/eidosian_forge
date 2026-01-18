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
def _test_wait_handle_invalid(self, tmpl, handle_name):
    self.stack = self.create_stack(template=tmpl)
    self.stack.create()
    rsrc = self.stack['wait_condition']
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    reason = rsrc.status_reason
    error_msg = 'ValueError: resources.wait_condition: %s is not a valid wait condition handle.' % handle_name
    self.assertEqual(reason, error_msg)