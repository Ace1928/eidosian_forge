import calendar
import time
import urllib
from cryptography.hazmat import backends
from cryptography.hazmat.primitives import serialization
from cryptography import x509 as cryptography_x509
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import service_token
from keystoneauth1 import session
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from castellan.common import exception
from castellan.common.objects import key as key_base_class
from castellan.common.objects import opaque_data as op_data
from castellan.i18n import _
from castellan.key_manager import key_manager
from barbicanclient import client as barbican_client_import
from barbicanclient import exceptions as barbican_exceptions
from oslo_utils import timeutils
def _get_active_order(self, barbican_client, order_ref):
    """Returns the order when it is active.

        Barbican key creation is done asynchronously, so this loop continues
        checking until the order is active or a timeout occurs.
        """
    active_status = 'ACTIVE'
    error_status = 'ERROR'
    number_of_retries = self.conf.barbican.number_of_retries
    retry_delay = self.conf.barbican.retry_delay
    order = barbican_client.orders.get(order_ref)
    time.sleep(0.25)
    for n in range(number_of_retries):
        if order.status == error_status:
            kwargs = {'status': error_status, 'code': order.error_status_code, 'reason': order.error_reason}
            msg = _('Order is in %(status)s status - status code: %(code)s, status reason: %(reason)s') % kwargs
            LOG.error(msg)
            raise exception.KeyManagerError(reason=msg)
        if order.status != active_status:
            kwargs = {'attempt': n, 'total': number_of_retries, 'status': order.status, 'active': active_status, 'delay': retry_delay}
            msg = _("Retry attempt #%(attempt)i out of %(total)i: Order status is '%(status)s'. Waiting for '%(active)s', will retry in %(delay)s seconds")
            LOG.info(msg, kwargs)
            time.sleep(retry_delay)
            order = barbican_client.orders.get(order_ref)
        else:
            return order
    msg = _("Exceeded retries: Failed to find '%(active)s' status within %(num_retries)i retries") % {'active': active_status, 'num_retries': number_of_retries}
    LOG.error(msg)
    raise exception.KeyManagerError(reason=msg)