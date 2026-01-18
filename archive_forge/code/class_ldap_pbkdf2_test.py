import logging
import warnings
from passlib import hash
from passlib.utils.compat import u
from passlib.tests.utils import TestCase, HandlerCase
from passlib.tests.test_handlers import UPASS_WAV
class ldap_pbkdf2_test(TestCase):

    def test_wrappers(self):
        """test ldap pbkdf2 wrappers"""
        self.assertTrue(hash.ldap_pbkdf2_sha1.verify('password', '{PBKDF2}1212$OB.dtnSEXZK8U5cgxU/GYQ$y5LKPOplRmok7CZp/aqVDVg8zGI'))
        self.assertTrue(hash.ldap_pbkdf2_sha256.verify('password', '{PBKDF2-SHA256}1212$4vjV83LKPjQzk31VI4E0Vw$hsYF68OiOUPdDZ1Fg.fJPeq1h/gXXY7acBp9/6c.tmQ'))
        self.assertTrue(hash.ldap_pbkdf2_sha512.verify('password', '{PBKDF2-SHA512}1212$RHY0Fr3IDMSVO/RSZyb5ow$eNLfBK.eVozomMr.1gYa17k9B7KIK25NOEshvhrSX.esqY3s.FvWZViXz4KoLlQI.BzY/YTNJOiKc5gBYFYGww'))