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
def _get_secret(self, context, object_id):
    """Returns the metadata of the secret.

        :param context: contains information of the user and the environment
                        for the request (castellan/context.py)
        :param object_id: UUID of the secret
        :return: the secret's metadata
        :raises HTTPAuthError: if object retrieval fails with 401
        :raises HTTPClientError: if object retrieval fails with 4xx
        :raises HTTPServerError: if object retrieval fails with 5xx
        """
    barbican_client = self._get_barbican_client(context)
    try:
        secret_ref = self._create_secret_ref(object_id)
        return barbican_client.secrets.get(secret_ref)
    except (barbican_exceptions.HTTPAuthError, barbican_exceptions.HTTPClientError, barbican_exceptions.HTTPServerError) as e:
        with excutils.save_and_reraise_exception():
            LOG.error('Error getting secret metadata: %s', e)