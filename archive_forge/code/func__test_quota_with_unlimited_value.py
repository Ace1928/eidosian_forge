from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder as c_plugin
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_quota_with_unlimited_value(self, prop_name):
    my_quota = self.stack['my_quota']
    props = self.stack.t.t['resources']['my_quota']['properties'].copy()
    props[prop_name] = -1
    my_quota.t = my_quota.t.freeze(properties=props)
    my_quota.reparse()
    my_quota.handle_create()
    kwargs = {'gigabytes': 5, 'snapshots': 2, 'volumes': 3}
    kwargs[prop_name] = -1
    self.quotas.update.assert_called_once_with('some_project_id', **kwargs)