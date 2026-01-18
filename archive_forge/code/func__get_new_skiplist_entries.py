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
def _get_new_skiplist_entries(self, properties, current_skiplist):
    insp = grouputils.GroupInspector.from_parent_resource(self)
    for r in properties.get(self.REMOVAL_POLICIES, []):
        if self.REMOVAL_RSRC_LIST in r:
            for n in r[self.REMOVAL_RSRC_LIST]:
                str_n = str(n)
                if str_n in current_skiplist or self.resource_id is None or str_n in insp.member_names(include_failed=True):
                    yield str_n
                elif isinstance(n, str):
                    try:
                        refids = self.get_output(self.REFS_MAP)
                    except (exception.NotFound, exception.TemplateOutputError) as op_err:
                        LOG.debug('Falling back to resource_by_refid()  due to %s', op_err)
                        rsrc = self.nested().resource_by_refid(n)
                        if rsrc is not None:
                            yield rsrc.name
                    else:
                        if refids is not None:
                            for name, refid in refids.items():
                                if refid == n:
                                    yield name
                                    break
    self._outputs = None