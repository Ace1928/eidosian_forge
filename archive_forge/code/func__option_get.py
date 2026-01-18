import http.client
import io
import logging
import math
import urllib.parse
from keystoneauth1.access import service_catalog as keystone_sc
from keystoneauth1 import identity as ks_identity
from keystoneauth1 import session as ks_session
from keystoneclient.v3 import client as ks_client
from oslo_config import cfg
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import units
import glance_store
from glance_store._drivers.swift import buffered
from glance_store._drivers.swift import connection_manager
from glance_store._drivers.swift import utils as sutils
from glance_store import capabilities
from glance_store.common import utils as gutils
from glance_store import driver
from glance_store import exceptions
from glance_store.i18n import _, _LE, _LI
from glance_store import location
def _option_get(self, param):
    if self.backend_group:
        result = getattr(getattr(self.conf, self.backend_group), param)
    else:
        result = getattr(self.conf.glance_store, param)
    if result is None:
        reason = _('Could not find %(param)s in configuration options.') % param
        LOG.error(reason)
        raise exceptions.BadStoreConfiguration(store_name='swift', reason=reason)
    return result