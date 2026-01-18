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
def _get_barbican_endpoint(self, auth, sess):
    if self.conf.barbican.barbican_endpoint:
        return self.conf.barbican.barbican_endpoint
    elif getattr(auth, 'service_catalog', None):
        endpoint_data = auth.service_catalog.endpoint_data_for(service_type='key-manager', interface=self.conf.barbican.barbican_endpoint_type, region_name=self.conf.barbican.barbican_region_name)
        return endpoint_data.url
    else:
        return auth.get_endpoint(sess, service_type='key-manager', interface=self.conf.barbican.barbican_endpoint_type, region_name=self.conf.barbican.barbican_region_name)