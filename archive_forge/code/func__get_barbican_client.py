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
def _get_barbican_client(self, context):
    """Creates a client to connect to the Barbican service.

        :param context: the user context for authentication
        :return: a Barbican Client object
        :raises Forbidden: if the context is None
        :raises KeyManagerError: if context is missing tenant or tenant is
                                 None or error occurs while creating client
        """
    if not context:
        msg = _('User is not authorized to use key manager.')
        LOG.error(msg)
        raise exception.Forbidden(msg)
    if self._barbican_client and self._current_context == context:
        return self._barbican_client
    try:
        auth = self._get_keystone_auth(context)
        verify_ssl = self.conf.barbican.verify_ssl
        verify_ssl_path = self.conf.barbican.verify_ssl_path
        verify = verify_ssl and verify_ssl_path or verify_ssl
        sess = session.Session(auth=auth, verify=verify)
        self._barbican_endpoint = self._get_barbican_endpoint(auth, sess)
        self._barbican_client = barbican_client_import.Client(session=sess, endpoint=self._barbican_endpoint)
        self._current_context = context
    except Exception as e:
        LOG.error('Error creating Barbican client: %s', e)
        raise exception.KeyManagerError(reason=e)
    self._base_url = self._create_base_url(auth, sess, self._barbican_endpoint)
    return self._barbican_client