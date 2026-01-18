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
def _assemble_for_rolling_update(self, total_capacity, max_updates, include_all=False, template_version=('heat_template_version', '2015-04-30')):
    names = list(self._resource_names(total_capacity))
    name_skiplist = self._name_skiplist()
    valid_resources = [(n, d) for n, d in grouputils.get_member_definitions(self) if n not in name_skiplist]
    targ_cap = self.get_size()

    def replace_priority(res_item):
        name, defn = res_item
        try:
            index = names.index(name)
        except ValueError:
            return 0
        else:
            if index < targ_cap:
                return targ_cap - index
            else:
                return total_capacity
    old_resources = sorted(valid_resources, key=replace_priority)
    existing_names = set((n for n, d in valid_resources))
    new_names = itertools.filterfalse(lambda n: n in existing_names, names)
    res_def = self.get_resource_def(include_all)
    definitions = scl_template.member_definitions(old_resources, res_def, total_capacity, max_updates, lambda: next(new_names), self.build_resource_definition)
    tmpl = scl_template.make_template(definitions, version=template_version)
    self._add_output_defns_to_template(tmpl, names)
    return tmpl