from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import keystone as k_plugin
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_invalid_property(self, prop_name):
    my_quota = self.stack['my_quota']
    props = self.stack.t.t['resources']['my_quota']['properties'].copy()
    props[prop_name] = -2
    my_quota.t = my_quota.t.freeze(properties=props)
    my_quota.reparse()
    error_msg = 'Property error: resources.my_quota.properties.%s: -2 is out of range (min: -1, max: None)' % prop_name
    self._test_validate(my_quota, error_msg)