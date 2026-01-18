import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def _add_output_defns_to_template(self, tmpl, resource_names):
    att_func = 'get_attr'
    get_attr = functools.partial(tmpl.functions[att_func], None, att_func)
    res_func = 'get_resource'
    get_res = functools.partial(tmpl.functions[res_func], None, res_func)
    for odefn in self._nested_output_defns(resource_names, get_attr, get_res):
        tmpl.add_output(odefn)