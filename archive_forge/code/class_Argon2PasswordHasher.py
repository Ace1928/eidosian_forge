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
class Argon2PasswordHasher(BasePasswordHasher):
    """
    Secure password hashing using the argon2 algorithm.

    This is the winner of the Password Hashing Competition 2013-2015
    (https://password-hashing.net). It requires the argon2-cffi library which
    depends on native C code and might cause portability issues.
    """
    algorithm = 'argon2'
    library = 'argon2'
    time_cost = 2
    memory_cost = 102400
    parallelism = 8

    def encode(self, password, salt):
        argon2 = self._load_library()
        params = self.params()
        data = argon2.low_level.hash_secret(password.encode(), salt.encode(), time_cost=params.time_cost, memory_cost=params.memory_cost, parallelism=params.parallelism, hash_len=params.hash_len, type=params.type)
        return self.algorithm + data.decode('ascii')

    def decode(self, encoded):
        argon2 = self._load_library()
        algorithm, rest = encoded.split('$', 1)
        assert algorithm == self.algorithm
        params = argon2.extract_parameters('$' + rest)
        variety, *_, b64salt, hash = rest.split('$')
        b64salt += '=' * (-len(b64salt) % 4)
        salt = base64.b64decode(b64salt).decode('latin1')
        return {'algorithm': algorithm, 'hash': hash, 'memory_cost': params.memory_cost, 'parallelism': params.parallelism, 'salt': salt, 'time_cost': params.time_cost, 'variety': variety, 'version': params.version, 'params': params}

    def verify(self, password, encoded):
        argon2 = self._load_library()
        algorithm, rest = encoded.split('$', 1)
        assert algorithm == self.algorithm
        try:
            return argon2.PasswordHasher().verify('$' + rest, password)
        except argon2.exceptions.VerificationError:
            return False

    def safe_summary(self, encoded):
        decoded = self.decode(encoded)
        return {_('algorithm'): decoded['algorithm'], _('variety'): decoded['variety'], _('version'): decoded['version'], _('memory cost'): decoded['memory_cost'], _('time cost'): decoded['time_cost'], _('parallelism'): decoded['parallelism'], _('salt'): mask_hash(decoded['salt']), _('hash'): mask_hash(decoded['hash'])}

    def must_update(self, encoded):
        decoded = self.decode(encoded)
        current_params = decoded['params']
        new_params = self.params()
        new_params.salt_len = current_params.salt_len
        update_salt = must_update_salt(decoded['salt'], self.salt_entropy)
        return current_params != new_params or update_salt

    def harden_runtime(self, password, encoded):
        pass

    def params(self):
        argon2 = self._load_library()
        return argon2.Parameters(type=argon2.low_level.Type.ID, version=argon2.low_level.ARGON2_VERSION, salt_len=argon2.DEFAULT_RANDOM_SALT_LENGTH, hash_len=argon2.DEFAULT_HASH_LENGTH, time_cost=self.time_cost, memory_cost=self.memory_cost, parallelism=self.parallelism)