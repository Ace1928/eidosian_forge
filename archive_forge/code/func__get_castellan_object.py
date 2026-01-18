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
def _get_castellan_object(self, secret, metadata_only=False):
    """Creates a Castellan managed object given the Barbican secret.

        The python barbicanclient lazy-loads the secret data, i.e. the secret
        data is not requested until secret.payload is called. If the user
        specifies metadata_only=True, secret.payload is never called,
        preventing unnecessary loading of secret data.

        :param secret: the barbican secret object
        :metadata_only: boolean indicating if the secret bytes should be
                        included in the managed object
        :returns: the castellan object
        """
    secret_type = op_data.OpaqueData
    for castellan_type, barbican_type in self._secret_type_dict.items():
        if barbican_type == secret.secret_type:
            secret_type = castellan_type
    if metadata_only:
        secret_data = None
    else:
        secret_data = self._get_secret_data(secret)
    if secret.secret_ref:
        object_id = self._retrieve_secret_uuid(secret.secret_ref)
    else:
        object_id = None
    if secret.created:
        time_stamp = timeutils.parse_isotime(str(secret.created)).timetuple()
        created = calendar.timegm(time_stamp)
    if issubclass(secret_type, key_base_class.Key):
        return secret_type(algorithm=secret.algorithm, bit_length=secret.bit_length, key=secret_data, name=secret.name, created=created, id=object_id, consumers=secret.consumers)
    else:
        return secret_type(secret_data, name=secret.name, created=created, id=object_id, consumers=secret.consumers)