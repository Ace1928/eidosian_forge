import collections
import copy
from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.manila import share as mshare
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def _init_share(self, stack_name):
    tmp = template_format.parse(manila_template)
    self.stack = utils.parse_stack(tmp, stack_name=stack_name)
    res_def = self.stack.t.resource_definitions(self.stack)['test_share']
    share = mshare.ManilaShare('test_share', res_def, self.stack)
    self.patchobject(share, 'data_set')
    mock_client = mock.MagicMock()
    client = mock.MagicMock(return_value=mock_client)
    share.client = client
    return share