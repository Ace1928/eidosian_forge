import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc5208
class PrivateKeyInfoTestCase(unittest.TestCase):
    pem_text = 'MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAUAwggE8AgEAAkEAx8CO8E0MNgEKXXDf\nI1xqBmQ+Gp3Srkqp45OApIu4lZ97n5VJ5HljU9wXcPIfx29Le3w8hCPEkugpLsdV\nGWx+EQIDAQABAkEAiv3f+DGEh6ddsPszKQXK+LuTwy2CRajKYgJnBxf5zpG50XK4\n899An+x/pGYVmVED1f0JCbk3BUbv7HViLq0qgQIhAOYlQJaQ8KJBijDpjF62lcVr\nQrqFPM4+ZrHsw0dVY2CZAiEA3jE5ngkVPfjFWEr7wS50EJhGiYlQeY4l+hADGIhd\nXDkCIQDIHt5xzmif/nOGop5/gS7ssp8ch1zfTh2IW4NWlOZMCQIgLZmYo5BlpaRK\njAZHiKwJ8eXuhAeEVo4PyTREDmLeFjECIQCfyUPDclPo2O8ycPpozwoGwvKFrNZJ\nVWRpRKqYnOAIXQ==\n'

    def setUp(self):
        self.asn1Spec = rfc5208.PrivateKeyInfo()

    def testDerCodec(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate