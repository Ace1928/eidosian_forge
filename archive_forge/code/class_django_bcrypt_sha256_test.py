from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import re
import warnings
from passlib import hash
from passlib.utils import repeat_string
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase, skipUnless, SkipTest
from passlib.tests.test_handlers import UPASS_USD, UPASS_TABLE
from passlib.tests.test_ext_django import DJANGO_VERSION, MIN_DJANGO_VERSION, \
from passlib.tests.test_handlers_argon2 import _base_argon2_test
@skipUnless(hash.bcrypt.has_backend(), 'no bcrypt backends available')
class django_bcrypt_sha256_test(HandlerCase, _DjangoHelper):
    """test django_bcrypt_sha256"""
    handler = hash.django_bcrypt_sha256
    forbidden_characters = None
    fuzz_salts_need_bcrypt_repair = True
    known_correct_hashes = [('', 'bcrypt_sha256$$2a$06$/3OeRpbOf8/l6nPPRdZPp.nRiyYqPobEZGdNRBWihQhiFDh1ws1tu'), (UPASS_LETMEIN, 'bcrypt_sha256$$2a$08$NDjSAIcas.EcoxCRiArvT.MkNiPYVhrsrnJsRkLueZOoV1bsQqlmC'), (UPASS_TABLE, 'bcrypt_sha256$$2a$06$kCXUnRFQptGg491siDKNTu8RxjBGSjALHRuvhPYNFsa4Ea5d9M48u'), (repeat_string('abc123', 72), 'bcrypt_sha256$$2a$06$Tg/oYyZTyAf.Nb3qSgN61OySmyXA8FoY4PjGizjE1QSDfuL5MXNni'), (repeat_string('abc123', 72) + 'qwr', 'bcrypt_sha256$$2a$06$Tg/oYyZTyAf.Nb3qSgN61Ocy0BEz1RK6xslSNi8PlaLX2pe7x/KQG'), (repeat_string('abc123', 72) + 'xyz', 'bcrypt_sha256$$2a$06$Tg/oYyZTyAf.Nb3qSgN61OvY2zoRVUa2Pugv2ExVOUT2YmhvxUFUa')]
    known_malformed_hashers = ['bcrypt_sha256$xyz$2a$06$/3OeRpbOf8/l6nPPRdZPp.nRiyYqPobEZGdNRBWihQhiFDh1ws1tu']

    def populate_settings(self, kwds):
        kwds.setdefault('rounds', 4)
        super(django_bcrypt_sha256_test, self).populate_settings(kwds)

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def random_rounds(self):
            return self.randintgauss(5, 8, 6, 1)

        def random_ident(self):
            return None