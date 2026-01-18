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
def _resolve_any_attribute(self, attr):
    """Method for resolving any attribute, including base attributes.

        This method uses basic _resolve_attribute method for resolving
        plugin-specific attributes. Base attributes will be resolved with
        corresponding method, which should be defined in each resource
        class.

        :param attr: attribute name, which will be resolved
        :returns: method of resource class, which resolve base attribute
        """
    if attr in self.base_attributes_schema:
        if self.resource_id is not None:
            with self._default_client_plugin().ignore_not_found:
                return getattr(self, '_{0}_resource'.format(attr))()
    else:
        with self._default_client_plugin().ignore_not_found:
            return self._resolve_attribute(attr)
    return None