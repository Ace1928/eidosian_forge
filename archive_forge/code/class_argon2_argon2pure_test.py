import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
class argon2_argon2pure_test(_base_argon2_test.create_backend_case('argon2pure')):
    handler = hash.argon2.using(memory_cost=32, parallelism=2)
    handler.pure_use_threads = True
    known_correct_hashes = _base_argon2_test.known_correct_hashes[:]
    known_correct_hashes.extend(((info['secret'], info['hash']) for info in reference_data if info['logM'] < 16))

    class FuzzHashGenerator(_base_argon2_test.FuzzHashGenerator):

        def random_rounds(self):
            return self.randintgauss(1, 3, 2, 1)