import binascii
import hashlib
import hmac
import re
from ntlm_auth.des import DES
def _ntowfv1(password):
    """
    [MS-NLMP] v28.0 2016-07-14

    3.3.1 NTLM v1 Authentication
    Same function as NTOWFv1 in document to create a one way hash of the
    password. Only used in NTLMv1 auth without session security

    :param password: The password or hash of the user we are trying to
        authenticate with
    :return digest: An NT hash of the password supplied
    """
    if re.match('^[a-fA-F\\d]{32}:[a-fA-F\\d]{32}$', password):
        nt_hash = binascii.unhexlify(password.split(':')[1])
        return nt_hash
    digest = hashlib.new('md4', password.encode('utf-16-le')).digest()
    return digest