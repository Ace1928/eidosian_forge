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
def _create_secret_ref(self, object_id):
    """Creates the URL required for accessing a secret.

        :param object_id: the UUID of the key to copy
        :return: the URL of the requested secret
        """
    if not object_id:
        msg = _('Key ID is None')
        raise exception.KeyManagerError(reason=msg)
    base_url = self._base_url
    if base_url[-1] != '/':
        base_url += '/'
    return urllib.parse.urljoin(base_url, 'secrets/' + object_id)