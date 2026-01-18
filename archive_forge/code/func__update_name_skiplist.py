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
def _update_name_skiplist(self, properties):
    """Resolve the remove_policies to names for removal."""
    curr_sl = set(self._current_skiplist())
    p_mode = properties.get(self.REMOVAL_POLICIES_MODE, self.REMOVAL_POLICY_APPEND)
    if p_mode == self.REMOVAL_POLICY_UPDATE:
        init_sl = set()
    else:
        init_sl = curr_sl
    updated_sl = init_sl | set(self._get_new_skiplist_entries(properties, curr_sl))
    if updated_sl != curr_sl:
        self.data_set('name_blacklist', ','.join(sorted(updated_sl)))