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
class CryptHash(BaseHash):

    def __init__(self, algorithm):
        super(CryptHash, self).__init__(algorithm)
        if not HAS_CRYPT:
            raise AnsibleError("crypt.crypt cannot be used as the 'crypt' python library is not installed or is unusable.", orig_exc=CRYPT_E)
        if sys.platform.startswith('darwin'):
            raise AnsibleError('crypt.crypt not supported on Mac OS X/Darwin, install passlib python module')
        if algorithm not in self.algorithms:
            raise AnsibleError("crypt.crypt does not support '%s' algorithm" % self.algorithm)
        display.deprecated('Encryption using the Python crypt module is deprecated. The Python crypt module is deprecated and will be removed from Python 3.13. Install the passlib library for continued encryption functionality.', version='2.17')
        self.algo_data = self.algorithms[algorithm]

    def hash(self, secret, salt=None, salt_size=None, rounds=None, ident=None):
        salt = self._salt(salt, salt_size)
        rounds = self._rounds(rounds)
        ident = self._ident(ident)
        return self._hash(secret, salt, rounds, ident)

    def _salt(self, salt, salt_size):
        salt_size = salt_size or self.algo_data.salt_size
        ret = salt or random_salt(salt_size)
        if re.search('[^./0-9A-Za-z]', ret):
            raise AnsibleError('invalid characters in salt')
        if self.algo_data.salt_exact and len(ret) != self.algo_data.salt_size:
            raise AnsibleError('invalid salt size')
        elif not self.algo_data.salt_exact and len(ret) > self.algo_data.salt_size:
            raise AnsibleError('invalid salt size')
        return ret

    def _rounds(self, rounds):
        if self.algorithm == 'bcrypt':
            return rounds or self.algo_data.implicit_rounds
        elif rounds == self.algo_data.implicit_rounds:
            return None
        else:
            return rounds

    def _ident(self, ident):
        if not ident:
            return self.algo_data.crypt_id
        if self.algorithm == 'bcrypt':
            return ident
        return None

    def _hash(self, secret, salt, rounds, ident):
        saltstring = ''
        if ident:
            saltstring = '$%s' % ident
        if rounds:
            if self.algorithm == 'bcrypt':
                saltstring += '$%d' % rounds
            else:
                saltstring += '$rounds=%d' % rounds
        saltstring += '$%s' % salt
        try:
            result = crypt.crypt(secret, saltstring)
            orig_exc = None
        except OSError as e:
            result = None
            orig_exc = e
        if not result:
            raise AnsibleError("crypt.crypt does not support '%s' algorithm" % self.algorithm, orig_exc=orig_exc)
        return result