import binascii
import hashlib
import hmac
import re
from ntlm_auth.des import DES
def _lmowfv1(password):
    """
    [MS-NLMP] v28.0 2016-07-14

    3.3.1 NTLM v1 Authentication
    Same function as LMOWFv1 in document to create a one way hash of the
    password. Only used in NTLMv1 auth without session security

    :param password: The password or hash of the user we are trying to
        authenticate with
    :return res: A Lan Manager hash of the password supplied
    """
    if re.match('^[a-fA-F\\d]{32}:[a-fA-F\\d]{32}$', password):
        lm_hash = binascii.unhexlify(password.split(':')[0])
        return lm_hash
    password = password.upper()
    lm_pw = password.encode('utf-8')
    padding_size = 0 if len(lm_pw) >= 14 else 14 - len(lm_pw)
    lm_pw += b'\x00' * padding_size
    magic_str = b'KGS!@#$%'
    res = b''
    dobj = DES(DES.key56_to_key64(lm_pw[0:7]))
    res += dobj.encrypt(magic_str)
    dobj = DES(DES.key56_to_key64(lm_pw[7:14]))
    res += dobj.encrypt(magic_str)
    return res