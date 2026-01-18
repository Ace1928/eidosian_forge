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
def _delete_stale_chunks(self, connection, container, chunk_list):
    for chunk in chunk_list:
        LOG.debug('Deleting chunk %s' % chunk)
        try:
            connection.delete_object(container, chunk)
        except Exception:
            msg = _('Failed to delete orphaned chunk %(container)s/%(chunk)s')
            LOG.exception(msg % {'container': container, 'chunk': chunk})