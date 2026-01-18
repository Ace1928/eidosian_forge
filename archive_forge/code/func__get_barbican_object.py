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
def _get_barbican_object(self, barbican_client, managed_object):
    """Converts the Castellan managed_object to a Barbican secret."""
    name = getattr(managed_object, 'name', None)
    try:
        algorithm = managed_object.algorithm
        bit_length = managed_object.bit_length
    except AttributeError:
        algorithm = None
        bit_length = None
    secret_type = self._secret_type_dict.get(type(managed_object), 'opaque')
    payload = self._get_normalized_payload(managed_object.get_encoded(), secret_type)
    secret = barbican_client.secrets.create(payload=payload, algorithm=algorithm, bit_length=bit_length, name=name, secret_type=secret_type)
    return secret