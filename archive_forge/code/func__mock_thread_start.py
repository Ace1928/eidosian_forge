from unittest import mock
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import template_format
from heat.engine import service
from heat.engine import stack
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _mock_thread_start(self, stack_id, func, *args, **kwargs):
    func(*args, **kwargs)
    return mock.Mock()