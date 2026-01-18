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
def _break_if_required(self, action, hook):
    """Block the resource until the hook is cleared if there is one."""
    if self.stack.env.registry.matches_hook(self.name, hook):
        self.trigger_hook(hook)
        self._add_event(self.action, self.status, '%(a)s paused until Hook %(h)s is cleared' % {'a': action, 'h': hook})
        LOG.info('Reached hook on %s', self)
        while self.has_hook(hook):
            try:
                yield
            except BaseException as exc:
                self.clear_hook(hook)
                self._add_event(self.action, self.status, 'Failure occurred while waiting.')
                if isinstance(exc, AssertionError) or not isinstance(exc, Exception):
                    raise