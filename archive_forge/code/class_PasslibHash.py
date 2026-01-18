from __future__ import (absolute_import, division, print_function)
import random
import re
import string
import sys
from collections import namedtuple
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.module_utils.six import text_type
from ansible.module_utils.common.text.converters import to_text, to_bytes
from ansible.utils.display import Display
class PasslibHash(BaseHash):

    def __init__(self, algorithm):
        super(PasslibHash, self).__init__(algorithm)
        if not PASSLIB_AVAILABLE:
            raise AnsibleError("passlib must be installed and usable to hash with '%s'" % algorithm, orig_exc=PASSLIB_E)
        display.vv("Using passlib to hash input with '%s'" % algorithm)
        try:
            self.crypt_algo = getattr(passlib.hash, algorithm)
        except Exception:
            raise AnsibleError("passlib does not support '%s' algorithm" % algorithm)

    def hash(self, secret, salt=None, salt_size=None, rounds=None, ident=None):
        salt = self._clean_salt(salt)
        rounds = self._clean_rounds(rounds)
        ident = self._clean_ident(ident)
        return self._hash(secret, salt=salt, salt_size=salt_size, rounds=rounds, ident=ident)

    def _clean_ident(self, ident):
        ret = None
        if not ident:
            if self.algorithm in self.algorithms:
                return self.algorithms.get(self.algorithm).implicit_ident
            return ret
        if self.algorithm == 'bcrypt':
            return ident
        return ret

    def _clean_salt(self, salt):
        if not salt:
            return None
        elif issubclass(self.crypt_algo.wrapped if isinstance(self.crypt_algo, PrefixWrapper) else self.crypt_algo, HasRawSalt):
            ret = to_bytes(salt, encoding='ascii', errors='strict')
        else:
            ret = to_text(salt, encoding='ascii', errors='strict')
        if self.algorithm == 'bcrypt':
            ret = bcrypt64.repair_unused(ret)
        return ret

    def _clean_rounds(self, rounds):
        algo_data = self.algorithms.get(self.algorithm)
        if rounds:
            return rounds
        elif algo_data and algo_data.implicit_rounds:
            return algo_data.implicit_rounds
        else:
            return None

    def _hash(self, secret, salt, salt_size, rounds, ident):
        settings = {}
        if salt:
            settings['salt'] = salt
        if salt_size:
            settings['salt_size'] = salt_size
        if rounds:
            settings['rounds'] = rounds
        if ident:
            settings['ident'] = ident
        try:
            if hasattr(self.crypt_algo, 'hash'):
                result = self.crypt_algo.using(**settings).hash(secret)
            elif hasattr(self.crypt_algo, 'encrypt'):
                result = self.crypt_algo.encrypt(secret, **settings)
            else:
                raise AnsibleError('installed passlib version %s not supported' % passlib.__version__)
        except ValueError as e:
            raise AnsibleError('Could not hash the secret.', orig_exc=e)
        if not result:
            raise AnsibleError("failed to hash with algorithm '%s'" % self.algorithm)
        return to_text(result, errors='strict')