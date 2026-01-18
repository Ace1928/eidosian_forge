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
def _delete_user_cred(self, stack_status=None, reason=None, raise_keystone_exception=False):
    if self.user_creds_id:
        user_creds = self._try_get_user_creds()
        if user_creds is not None:
            trust_id = user_creds.get('trust_id')
            if trust_id:
                try:
                    trustor_id = user_creds.get('trustor_user_id')
                    if self.context.user_id != trustor_id:
                        LOG.debug("Context user_id doesn't match trustor, using stored context")
                        sc = self.stored_context()
                        sc.clients.client('keystone').delete_trust(trust_id)
                    else:
                        self.clients.client('keystone').delete_trust(trust_id)
                except Exception:
                    LOG.exception('Error deleting trust')
                    if raise_keystone_exception:
                        raise
        try:
            ucreds_object.UserCreds.delete(self.context, self.user_creds_id)
        except exception.NotFound:
            LOG.info('Tried to delete user_creds that do not exist (stack=%(stack)s user_creds_id=%(uc)s)', {'stack': self.id, 'uc': self.user_creds_id})
        self.user_creds_id = None
    return (stack_status, reason)