from datetime import datetime
from datetime import timedelta
from unittest import mock
from oslo_config import cfg
from heat.common import template_format
from heat.engine import environment
from heat.engine import stack as parser
from heat.engine import template as templatem
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_objects
from heat.objects import stack as stack_object
from heat.objects import sync_point as sync_point_object
from heat.rpc import worker_client
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import utils
def _fake_db_resources(self, stack):
    db_resources = {}
    i = 0
    for rsrc_name in ['E', 'D', 'C', 'B', 'A']:
        i += 1
        rsrc = mock.MagicMock()
        rsrc.id = i
        rsrc.name = rsrc_name
        rsrc.current_template_id = stack.prev_raw_template_id
        db_resources[i] = rsrc
    db_resources[3].requires = [4, 5]
    db_resources[1].requires = [3]
    db_resources[2].requires = [3]
    return db_resources