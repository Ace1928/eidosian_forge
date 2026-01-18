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
def _handle_repl_val(self, res_name, val):
    repl_var = self.properties[self.INDEX_VAR]

    def recurse(x):
        return self._handle_repl_val(res_name, x)
    if isinstance(val, str):
        return val.replace(repl_var, res_name)
    elif isinstance(val, collections.abc.Mapping):
        return {k: recurse(v) for k, v in val.items()}
    elif isinstance(val, collections.abc.Sequence):
        return [recurse(v) for v in val]
    return val