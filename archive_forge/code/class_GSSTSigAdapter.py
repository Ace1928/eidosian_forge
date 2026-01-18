import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
class GSSTSigAdapter:

    def __init__(self, keyring):
        self.keyring = keyring

    def __call__(self, message, keyname):
        if keyname in self.keyring:
            key = self.keyring[keyname]
            if isinstance(key, Key) and key.algorithm == GSS_TSIG:
                if message:
                    GSSTSigAdapter.parse_tkey_and_step(key, message, keyname)
            return key
        else:
            return None

    @classmethod
    def parse_tkey_and_step(cls, key, message, keyname):
        try:
            rrset = message.find_rrset(message.answer, keyname, dns.rdataclass.ANY, dns.rdatatype.TKEY)
            if rrset:
                token = rrset[0].key
                gssapi_context = key.secret
                return gssapi_context.step(token)
        except KeyError:
            pass