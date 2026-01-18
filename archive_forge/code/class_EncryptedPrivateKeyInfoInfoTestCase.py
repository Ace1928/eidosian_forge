import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc5208
class EncryptedPrivateKeyInfoInfoTestCase(unittest.TestCase):
    pem_text = 'MIIBgTAbBgkqhkiG9w0BBQMwDgQIdtFgDWnipT8CAggABIIBYN0hkm2xqkTCt8dJ\niZS8+HNiyHxy8g+rmWSXv/i+bTHFUReZA2GINtTRUkWpXqWcSHxNslgf7QdfgbVJ\nxQiUM+lLhwOFh85iAHR3xmPU1wfN9NvY9DiLSpM0DMhF3OvAMZD75zIhA0GSKu7w\ndUu7ey7H4fv7bez6RhEyLdKw9/Lf2KNStNOs4ow9CAtCoxeoMSniTt6CNhbvCkve\n9vNHKiGavX1tS/YTog4wiiGzh2YxuW1RiQpTdhWiKyECgD8qQVg2tY5t3QRcXrzi\nOkStpkiAPAbiwS/gyHpsqiLo0al63SCxRefugbn1ucZyc5Ya59e3xNFQXCNhYl+Z\nHl3hIl3cssdWZkJ455Z/bBE29ks1HtsL+bTfFi+kw/4yuMzoaB8C7rXScpGNI/8E\npvTU2+wtuoOFcttJregtR94ZHu5wgdYqRydmFNG8PnvZT1mRMmQgUe/vp88FMmsZ\ndLsZjNQ=\n'

    def setUp(self):
        self.asn1Spec = rfc5208.EncryptedPrivateKeyInfo()

    def testDerCodec(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate