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
def _prepare_update_props(self, after, before):
    before_props = before.properties(self.properties_schema, self.context)
    self.regenerate_info_schema(after)
    after.set_translation_rules(self.translation_rules(self.properties))
    after_props = after.properties(self.properties_schema, self.context)
    self.translate_properties(after_props)
    self.translate_properties(before_props, ignore_resolve_error=True)
    if (cfg.CONF.observe_on_update or self.converge) and before_props:
        if not self.resource_id:
            raise UpdateReplace(self)
        try:
            resource_reality = self.get_live_state(before_props)
            if resource_reality:
                self._update_properties_with_live_state(before_props, resource_reality)
        except exception.EntityNotFound:
            raise UpdateReplace(self)
        except Exception as ex:
            LOG.warning("Resource cannot be updated with it's live state in case of next error: %s", ex)
    return (after_props, before_props)