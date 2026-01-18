import collections
import random
import time
from neutron_lib._i18n import _
from neutron_lib import context
from neutron_lib import exceptions
from neutron_lib.utils import runtime
from oslo_config import cfg
from oslo_log import log as logging
import oslo_messaging
from oslo_messaging import exceptions as oslomsg_exc
from oslo_messaging import serializer as om_serializer
from oslo_service import service
from oslo_utils import excutils
from osprofiler import profiler
class _ContextWrapper(object):

    def __init__(self, original_context):
        self._original_context = original_context

    def __getattr__(self, name):
        return getattr(self._original_context, name)

    def cast(self, ctxt, method, **kwargs):
        try:
            self._original_context.cast(ctxt, method, **kwargs)
        except oslomsg_exc.MessageDeliveryFailure as e:
            LOG.debug('Ignored exception during cast: %s', str(e))