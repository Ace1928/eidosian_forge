import base64
import hashlib
import hmac
import struct
import dns.exception
import dns.name
import dns.rcode
import dns.rdataclass
class GSSTSig:
    """
    GSS-TSIG TSIG implementation.  This uses the GSS-API context established
    in the TKEY message handshake to sign messages using GSS-API message
    integrity codes, per the RFC.

    In order to avoid a direct GSSAPI dependency, the keyring holds a ref
    to the GSSAPI object required, rather than the key itself.
    """

    def __init__(self, gssapi_context):
        self.gssapi_context = gssapi_context
        self.data = b''
        self.name = 'gss-tsig'

    def update(self, data):
        self.data += data

    def sign(self):
        return self.gssapi_context.get_signature(self.data)

    def verify(self, expected):
        try:
            return self.gssapi_context.verify_signature(self.data, expected)
        except Exception:
            raise BadSignature