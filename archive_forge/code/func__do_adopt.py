from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.engine import service
from heat.engine import stack as parser
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import utils
def _do_adopt(self, stack_name, template, input_params, adopt_data):
    result = self.man.create_stack(self.ctx, stack_name, template, input_params, None, {'adopt_stack_data': str(adopt_data)})
    self.man.thread_group_mgr.stop(result['stack_id'], graceful=True)
    return result