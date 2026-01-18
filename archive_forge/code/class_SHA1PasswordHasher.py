import base64
import binascii
import functools
import hashlib
import importlib
import math
import warnings
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.core.signals import setting_changed
from django.dispatch import receiver
from django.utils.crypto import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.module_loading import import_string
from django.utils.translation import gettext_noop as _
class SHA1PasswordHasher(BasePasswordHasher):
    """
    The SHA1 password hashing algorithm (not recommended)
    """
    algorithm = 'sha1'

    def __init__(self, *args, **kwargs):
        warnings.warn('django.contrib.auth.hashers.SHA1PasswordHasher is deprecated.', RemovedInDjango51Warning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def encode(self, password, salt):
        self._check_encode_args(password, salt)
        hash = hashlib.sha1((salt + password).encode()).hexdigest()
        return '%s$%s$%s' % (self.algorithm, salt, hash)

    def decode(self, encoded):
        algorithm, salt, hash = encoded.split('$', 2)
        assert algorithm == self.algorithm
        return {'algorithm': algorithm, 'hash': hash, 'salt': salt}

    def verify(self, password, encoded):
        decoded = self.decode(encoded)
        encoded_2 = self.encode(password, decoded['salt'])
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {_('algorithm'): decoded['algorithm'], _('salt'): mask_hash(decoded['salt'], show=2), _('hash'): mask_hash(decoded['hash'])}

    def must_update(self, encoded):
        decoded = self.decode(encoded)
        return must_update_salt(decoded['salt'], self.salt_entropy)

    def harden_runtime(self, password, encoded):
        pass