import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import template as cfn_template
from heat.engine import function
from heat.engine.hot import functions as hot_funcs
from heat.engine.hot import parameters
from heat.engine import rsrc_defn
from heat.engine import template_common
def _rsrc_defn_args(self, stack, name, data):
    for arg in super(HOTemplate20161014, self)._rsrc_defn_args(stack, name, data):
        yield arg
    parse = functools.partial(self.parse, stack)
    parse_cond = functools.partial(self.parse_condition, stack)
    yield ('external_id', self._parse_resource_field(self.RES_EXTERNAL_ID, (str, function.Function), 'string', name, data, parse))
    yield ('condition', self._parse_resource_field(self.RES_CONDITION, (str, bool, function.Function), 'string_or_boolean', name, data, parse_cond))