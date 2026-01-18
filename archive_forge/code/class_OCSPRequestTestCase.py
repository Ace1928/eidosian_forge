import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc2560
class OCSPRequestTestCase(unittest.TestCase):
    pem_text = 'MGowaDBBMD8wPTAJBgUrDgMCGgUABBS3ZrMV9C5Dko03aH13cEZeppg3wgQUkqR1LKSevoFE63n8\nisWVpesQdXMCBDXe9M+iIzAhMB8GCSsGAQUFBzABAgQSBBBjdJOiIW9EKJGELNNf/rdA\n'

    def setUp(self):
        self.asn1Spec = rfc2560.OCSPRequest()

    def testDerCodec(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate