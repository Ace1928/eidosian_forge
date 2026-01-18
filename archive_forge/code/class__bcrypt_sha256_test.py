from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
import warnings
from passlib import hash
from passlib.handlers.bcrypt import IDENT_2, IDENT_2X
from passlib.utils import repeat_string, to_bytes, is_safe_crypt_input
from passlib.utils.compat import irange, PY3
from passlib.tests.utils import HandlerCase, TEST_MODE
from passlib.tests.test_handlers import UPASS_TABLE
class _bcrypt_sha256_test(HandlerCase):
    """base for BCrypt-SHA256 test cases"""
    handler = hash.bcrypt_sha256
    reduce_default_rounds = True
    forbidden_characters = None
    fuzz_salts_need_bcrypt_repair = True
    known_correct_hashes = [('', '$bcrypt-sha256$2a,5$E/e/2AOhqM5W/KJTFQzLce$F6dYSxOdAEoJZO2eoHUZWZljW/e0TXO'), ('password', '$bcrypt-sha256$2a,5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu'), (UPASS_TABLE, '$bcrypt-sha256$2a,5$.US1fQ4TQS.ZTz/uJ5Kyn.$QNdPDOTKKT5/sovNz1iWg26quOU4Pje'), (UPASS_TABLE.encode('utf-8'), '$bcrypt-sha256$2a,5$.US1fQ4TQS.ZTz/uJ5Kyn.$QNdPDOTKKT5/sovNz1iWg26quOU4Pje'), ('password', '$bcrypt-sha256$2b,5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu'), (UPASS_TABLE, '$bcrypt-sha256$2b,5$.US1fQ4TQS.ZTz/uJ5Kyn.$QNdPDOTKKT5/sovNz1iWg26quOU4Pje'), (repeat_string('abc123', 72), '$bcrypt-sha256$2b,5$X1g1nh3g0v4h6970O68cxe$r/hyEtqJ0teqPEmfTLoZ83ciAI1Q74.'), (repeat_string('abc123', 72) + 'qwr', '$bcrypt-sha256$2b,5$X1g1nh3g0v4h6970O68cxe$021KLEif6epjot5yoxk0m8I0929ohEa'), (repeat_string('abc123', 72) + 'xyz', '$bcrypt-sha256$2b,5$X1g1nh3g0v4h6970O68cxe$7.1kgpHduMGEjvM3fX6e/QCvfn6OKja'), ('', '$bcrypt-sha256$v=2,t=2b,r=5$E/e/2AOhqM5W/KJTFQzLce$WFPIZKtDDTriqWwlmRFfHiOTeheAZWe'), ('password', '$bcrypt-sha256$v=2,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe$wOK1VFFtS8IGTrGa7.h5fs0u84qyPbS'), (UPASS_TABLE, '$bcrypt-sha256$v=2,t=2b,r=5$.US1fQ4TQS.ZTz/uJ5Kyn.$pzzgp40k8reM1CuQb03PvE0IDPQSdV6'), (UPASS_TABLE.encode('utf-8'), '$bcrypt-sha256$v=2,t=2b,r=5$.US1fQ4TQS.ZTz/uJ5Kyn.$pzzgp40k8reM1CuQb03PvE0IDPQSdV6'), (repeat_string('abc123', 72), '$bcrypt-sha256$v=2,t=2b,r=5$X1g1nh3g0v4h6970O68cxe$zu1cloESVFIOsUIo7fCEgkdHaI9SSue'), (repeat_string('abc123', 72) + 'qwr', '$bcrypt-sha256$v=2,t=2b,r=5$X1g1nh3g0v4h6970O68cxe$CBF9csfEdW68xv3DwE6xSULXMtqEFP.'), (repeat_string('abc123', 72) + 'xyz', '$bcrypt-sha256$v=2,t=2b,r=5$X1g1nh3g0v4h6970O68cxe$zC/1UDUG2ofEXB6Onr2vvyFzfhEOS3S')]
    known_correct_configs = [('$bcrypt-sha256$2a,5$5Hg1DKFqPE8C2aflZ5vVoe', 'password', '$bcrypt-sha256$2a,5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu'), ('$bcrypt-sha256$v=2,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe', 'password', '$bcrypt-sha256$v=2,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe$wOK1VFFtS8IGTrGa7.h5fs0u84qyPbS')]
    known_malformed_hashes = ['$bcrypt-sha256$2a,5$5Hg1DKF!PE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$2c,5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$2x,5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$2a,05$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$2a,5$5Hg1DKFqPE8C2aflZ5vVoe$', '$bcrypt-sha256$v=2,t=2b,r=5$5Hg1DKF!PE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=1,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=3,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=2,t=2c,r=5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=2,t=2a,r=5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=2,t=2x,r=5$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=2,t=2b,r=05$5Hg1DKFqPE8C2aflZ5vVoe$12BjNE0p7axMg55.Y/mHsYiVuFBDQyu', '$bcrypt-sha256$v=2,t=2b,r=5$5Hg1DKFqPE8C2aflZ5vVoe$']

    def setUp(self):
        if TEST_MODE('full') and self.backend == 'builtin':
            key = 'PASSLIB_BUILTIN_BCRYPT'
            orig = os.environ.get(key)
            if orig:
                self.addCleanup(os.environ.__setitem__, key, orig)
            else:
                self.addCleanup(os.environ.__delitem__, key)
            os.environ[key] = 'enabled'
        super(_bcrypt_sha256_test, self).setUp()
        warnings.filterwarnings('ignore', '.*backend is vulnerable to the bsd wraparound bug.*')

    def populate_settings(self, kwds):
        if self.backend == 'builtin':
            kwds.setdefault('rounds', 4)
        super(_bcrypt_sha256_test, self).populate_settings(kwds)

    def require_many_idents(self):
        raise self.skipTest('multiple idents not supported')

    def test_30_HasOneIdent(self):
        handler = self.handler
        handler(use_defaults=True)
        self.assertRaises(ValueError, handler, ident='$2y$', use_defaults=True)

    class FuzzHashGenerator(HandlerCase.FuzzHashGenerator):

        def random_rounds(self):
            return self.randintgauss(5, 8, 6, 1)

        def random_ident(self):
            return '2b'

    def test_using_version(self):
        handler = self.handler
        self.assertEqual(handler.version, 2)
        subcls = handler.using(version=1)
        self.assertEqual(subcls.version, 1)
        self.assertRaises(ValueError, handler.using, version=999)
        subcls = handler.using(version=1, ident='2a')
        self.assertRaises(ValueError, handler.using, ident='2a')

    def test_calc_digest_v2(self):
        """
        test digest calc v2 matches bcrypt()
        """
        from passlib.hash import bcrypt
        from passlib.crypto.digest import compile_hmac
        from passlib.utils.binary import b64encode
        salt = 'nyKYxTAvjmy6lMDYMl11Uu'
        secret = 'test'
        temp_digest = compile_hmac('sha256', salt.encode('ascii'))(secret.encode('ascii'))
        temp_digest = b64encode(temp_digest).decode('ascii')
        self.assertEqual(temp_digest, 'J5TlyIDm+IcSWmKiDJm+MeICndBkFVPn4kKdJW8f+xY=')
        bcrypt_digest = bcrypt(ident='2b', salt=salt, rounds=12)._calc_checksum(temp_digest)
        self.assertEqual(bcrypt_digest, 'M0wE0Ov/9LXoQFCe.jRHu3MSHPF54Ta')
        self.assertTrue(bcrypt.verify(temp_digest, '$2b$12$' + salt + bcrypt_digest))
        result = self.handler(ident='2b', salt=salt, rounds=12)._calc_checksum(secret)
        self.assertEqual(result, bcrypt_digest)