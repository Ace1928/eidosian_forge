import collections
import contextlib
import datetime as dt
import itertools
import pydoc
import re
import tenacity
import weakref
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import short_id
from heat.common import timeutils
from heat.engine import attributes
from heat.engine.cfn import template as cfn_tmpl
from heat.engine import clients
from heat.engine.clients import default_client_plugin
from heat.engine import environment
from heat.engine import event
from heat.engine import function
from heat.engine.hot import template as hot_tmpl
from heat.engine import node_data
from heat.engine import properties
from heat.engine import resources
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import status
from heat.engine import support
from heat.engine import sync_point
from heat.engine import template
from heat.objects import resource as resource_objects
from heat.objects import resource_data as resource_data_objects
from heat.objects import resource_properties_data as rpd_objects
from heat.rpc import client as rpc_client
def _prepare_update_replace_handler(self, action):
    """Return the handler method for preparing to replace a resource.

        This may be either restore_prev_rsrc() (in the case of a legacy
        rollback) or, more typically, prepare_for_replace().

        If the plugin has not overridden the method, then None is returned in
        place of the default method (which is empty anyway).
        """
    if self.stack.action == 'ROLLBACK' and self.stack.status == 'IN_PROGRESS' and (not self.stack.convergence):
        if self.restore_prev_rsrc != Resource.restore_prev_rsrc:
            return self.restore_prev_rsrc
    elif self.prepare_for_replace != Resource.prepare_for_replace:
        return self.prepare_for_replace
    return None