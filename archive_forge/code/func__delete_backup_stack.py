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
def _delete_backup_stack(self, stack):

    def failed(child):
        return child.action == child.CREATE and child.status in (child.FAILED, child.IN_PROGRESS)

    def copy_data(source_res, destination_res):
        if source_res.data():
            for key, val in source_res.data().items():
                destination_res.data_set(key, val)
    for key, backup_res in stack.resources.items():
        backup_res_id = backup_res.resource_id
        curr_res = self.resources.get(key)
        if backup_res_id is not None and curr_res is not None:
            curr_res_id = curr_res.resource_id
            if any((failed(child) for child in self.dependencies[curr_res])) or curr_res.status in (curr_res.FAILED, curr_res.IN_PROGRESS):
                self.resources[key].resource_id = backup_res_id
                self.resources[key].properties = backup_res.properties
                copy_data(backup_res, self.resources[key])
                stack.resources[key].resource_id = curr_res_id
                stack.resources[key].properties = curr_res.properties
                copy_data(curr_res, stack.resources[key])
    stack.delete(backup=True)