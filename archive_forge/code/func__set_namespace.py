import binascii
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import Encoding
from cryptography.hazmat.primitives.serialization import NoEncryption
from cryptography.hazmat.primitives.serialization import PrivateFormat
from cryptography.hazmat.primitives.serialization import PublicFormat
import os
import time
import uuid
from keystoneauth1 import loading
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
import requests
from castellan.common import exception
from castellan.common.objects import private_key as pri_key
from castellan.common.objects import public_key as pub_key
from castellan.common.objects import symmetric_key as sym_key
from castellan.i18n import _
from castellan.key_manager import key_manager
def _set_namespace(self, headers):
    if self._namespace:
        headers['X-Vault-Namespace'] = self._namespace
    return headers