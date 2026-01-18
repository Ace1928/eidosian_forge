import collections
import contextlib
import copy
import eventlet
import functools
import re
import warnings
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import timeutils as oslo_timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context as common_context
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import lifecycle_plugin_utils
from heat.engine import api
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import event
from heat.engine.notification import stack as notification
from heat.engine import parameter_groups as param_groups
from heat.engine import parent_rsrc
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.engine import status
from heat.engine import stk_defn
from heat.engine import sync_point
from heat.engine import template as tmpl
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_worker_client
def _resource_from_db_resource(self, db_res, stk_def_cache=None):
    tid = db_res.current_template_id
    if tid is None:
        tid = self.t.id
    if tid == self.t.id:
        cur_res = self.resources.get(db_res.name)
        if cur_res is not None and cur_res.id == db_res.id:
            return cur_res
        stk_def = self.defn
    elif stk_def_cache and tid in stk_def_cache:
        stk_def = stk_def_cache[tid]
    else:
        try:
            t = tmpl.Template.load(self.context, tid)
        except exception.NotFound:
            return None
        stk_def = self.defn.clone_with_new_template(t, self.identifier())
        if stk_def_cache is not None:
            stk_def_cache[tid] = stk_def
    try:
        defn = stk_def.resource_definition(db_res.name)
    except KeyError:
        return None
    with self._previous_definition(stk_def):
        res = resource.Resource(db_res.name, defn, self)
        res._load_data(db_res)
    return res