import binascii
import hashlib
import hmac
import re
from ntlm_auth.des import DES

    [MS-NLMP] v28.0 2016-07-14

    3.3.2 NTLM v2 Authentication
    Same function as NTOWFv2 (and LMOWFv2) in document to create a one way hash
    of the password. This combines some extra security features over the v1
    calculations used in NTLMv2 auth.

    :param user_name: The user name of the user we are trying to authenticate
        with
    :param password: The password of the user we are trying to authenticate
        with
    :param domain_name: The domain name of the user account we are
        authenticated with
    :return digest: An NT hash of the parameters supplied
    