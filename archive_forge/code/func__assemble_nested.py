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
def _assemble_nested(self, names, include_all=False, template_version=('heat_template_version', '2015-04-30')):
    def_dict = self.get_resource_def(include_all)
    definitions = [(k, self.build_resource_definition(k, def_dict)) for k in names]
    tmpl = scl_template.make_template(definitions, version=template_version)
    self._add_output_defns_to_template(tmpl, [k for k, d in definitions])
    return tmpl