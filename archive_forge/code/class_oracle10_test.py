from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import sys
import warnings
from passlib import exc, hash
from passlib.utils import repeat_string
from passlib.utils.compat import irange, PY3, u, get_method_function
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, \
class oracle10_test(UserHandlerMixin, HandlerCase):
    handler = hash.oracle10
    secret_case_insensitive = True
    user_case_insensitive = True
    known_correct_hashes = [(('tiger', 'scott'), 'F894844C34402B67'), ((u('ttTiGGeR'), u('ScO')), '7AA1A84E31ED7771'), (('d_syspw', 'SYSTEM'), '1B9F1F9A5CB9EB31'), (('strat_passwd', 'strat_user'), 'AEBEDBB4EFB5225B'), (('#95LWEIGHTS', 'USER'), '000EA4D72A142E29'), (('CIAO2010', 'ALFREDO'), 'EB026A76F0650F7B'), (('GLOUGlou', 'Bob'), 'CDC6B483874B875B'), (('GLOUGLOUTER', 'bOB'), 'EF1F9139DB2D5279'), (('LONG_MOT_DE_PASSE_OUI', 'BOB'), 'EC8147ABB3373D53'), ((UPASS_TABLE, 'System'), 'B915A853F297B281')]
    known_unidentified_hashes = ['F894844C34402B6Z']