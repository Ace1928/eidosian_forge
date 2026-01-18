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
def _nested_output_defns(self, resource_names, get_attr_fn, get_res_fn):
    for attr in self.referenced_attrs():
        if isinstance(attr, str):
            key, path = (attr, [])
        else:
            key, path = (attr[0], list(attr[1:]))
        output_name = self._attribute_output_name(key, *path)
        value = None
        if key.startswith('resource.'):
            keycomponents = key.split('.', 2)
            res_name = keycomponents[1]
            attr_path = keycomponents[2:] + path
            if attr_path:
                if res_name in resource_names:
                    value = get_attr_fn([res_name] + attr_path)
            else:
                output_name = key = self.REFS_MAP
        elif key == self.ATTR_ATTRIBUTES and path:
            value = {r: get_attr_fn([r] + path) for r in resource_names}
        elif key not in self.ATTRIBUTES:
            value = [get_attr_fn([r, key] + path) for r in resource_names]
        if key == self.REFS:
            value = [get_res_fn(r) for r in resource_names]
        if value is not None:
            yield output.OutputDefinition(output_name, value)
    value = {r: get_res_fn(r) for r in resource_names}
    yield output.OutputDefinition(self.REFS_MAP, value)