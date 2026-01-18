import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
def _retry_on_internal_server_error(e):
    if isinstance(e, apiclient_exception.InternalServerError):
        return True
    return False