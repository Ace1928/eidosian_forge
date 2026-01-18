import logging
from cinderclient.apiclient import exceptions as apiclient_exception
from cinderclient import exceptions as cinder_exception
from keystoneauth1 import exceptions as keystone_exc
from oslo_utils import excutils
import retrying
from glance_store import exceptions
from glance_store.i18n import _LE
def _retry_on_bad_request(e):
    if isinstance(e, cinder_exception.BadRequest):
        return True
    return False