import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
class _FakeVerifier(object):
    name = 'verifier'