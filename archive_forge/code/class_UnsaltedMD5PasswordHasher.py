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
class UnsaltedMD5PasswordHasher(BasePasswordHasher):
    """
    Incredibly insecure algorithm that you should *never* use; stores unsalted
    MD5 hashes without the algorithm prefix, also accepts MD5 hashes with an
    empty salt.

    This class is implemented because Django used to store passwords this way
    and to accept such password hashes. Some older Django installs still have
    these values lingering around so we need to handle and upgrade them
    properly.
    """
    algorithm = 'unsalted_md5'

    def __init__(self, *args, **kwargs):
        warnings.warn('django.contrib.auth.hashers.UnsaltedMD5PasswordHasher is deprecated.', RemovedInDjango51Warning, stacklevel=2)
        super().__init__(*args, **kwargs)

    def salt(self):
        return ''

    def encode(self, password, salt):
        if salt != '':
            raise ValueError('salt must be empty.')
        return hashlib.md5(password.encode()).hexdigest()

    def decode(self, encoded):
        return {'algorithm': self.algorithm, 'hash': encoded, 'salt': None}

    def verify(self, password, encoded):
        if len(encoded) == 37:
            encoded = encoded.removeprefix('md5$$')
        encoded_2 = self.encode(password, '')
        return constant_time_compare(encoded, encoded_2)

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {_('algorithm'): decoded['algorithm'], _('hash'): mask_hash(decoded['hash'], show=3)}

    def harden_runtime(self, password, encoded):
        pass