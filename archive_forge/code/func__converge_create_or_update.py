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
@reset_state_on_error
def _converge_create_or_update(self, pre_converge=None):
    current_resources = self._update_or_store_resources()
    self._compute_convg_dependencies(self.ext_rsrcs_db, self.dependencies, current_resources)
    self.current_deps = {'edges': [[rqr, rqd] for rqr, rqd in self.convergence_dependencies.graph().edges()]}
    stack_id = self.store()
    if stack_id is None:
        LOG.warning('Failed to store stack %(name)s with traversal ID %(trvsl_id)s, aborting stack %(action)s', {'name': self.name, 'trvsl_id': self.current_traversal, 'action': self.action})
        return
    if callable(pre_converge):
        pre_converge()
    if self.action == self.DELETE:
        try:
            self.delete_all_snapshots()
        except Exception as exc:
            self.state_set(self.action, self.FAILED, str(exc))
            self.purge_db()
            return
    LOG.debug('Starting traversal %s with dependencies: %s', self.current_traversal, self.convergence_dependencies)
    for rsrc_id, is_update in self.convergence_dependencies:
        sync_point.create(self.context, rsrc_id, self.current_traversal, is_update, self.id)
    sync_point.create(self.context, self.id, self.current_traversal, True, self.id)
    leaves = set(self.convergence_dependencies.leaves())
    if not leaves:
        self.mark_complete()
    else:
        for rsrc_id, is_update in sorted(leaves, key=lambda n: n.is_update):
            if is_update:
                LOG.info('Triggering resource %s for update', rsrc_id)
            else:
                LOG.info('Triggering resource %s for cleanup', rsrc_id)
            input_data = sync_point.serialize_input_data({})
            self.worker_client.check_resource(self.context, rsrc_id, self.current_traversal, input_data, is_update, self.adopt_stack_data, self.converge)
            if scheduler.ENABLE_SLEEP:
                eventlet.sleep(1)