import logging
import re
import warnings
from passlib import hash
from passlib.utils.compat import unicode
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE, PASS_TABLE_UTF8
class FuzzHashGenerator(_base_argon2_test.FuzzHashGenerator):

    def random_rounds(self):
        return self.randintgauss(1, 3, 2, 1)