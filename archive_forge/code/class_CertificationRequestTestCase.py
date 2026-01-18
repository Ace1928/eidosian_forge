import sys
from pyasn1.codec.der import decoder as der_decoder
from pyasn1.codec.der import encoder as der_encoder
from pyasn1_modules import pem
from pyasn1_modules import rfc2314
class CertificationRequestTestCase(unittest.TestCase):
    pem_text = 'MIIDATCCAekCAQAwgZkxCzAJBgNVBAYTAlJVMRYwFAYDVQQIEw1Nb3Njb3cgUmVn\naW9uMQ8wDQYDVQQHEwZNb3Njb3cxGjAYBgNVBAoTEVNOTVAgTGFib3JhdG9yaWVz\nMQwwCgYDVQQLFANSJkQxFTATBgNVBAMTDHNubXBsYWJzLmNvbTEgMB4GCSqGSIb3\nDQEJARYRaW5mb0Bzbm1wbGFicy5jb20wggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAw\nggEKAoIBAQC9n2NfGS98JDBmAXQn+vNUyPB3QPYC1cwpX8UMYh9MdAmBZJCnvXrQ\nPp14gNAv6AQKxefmGES1b+Yd+1we9HB8AKm1/8xvRDUjAvy4iO0sqFCPvIfSujUy\npBcfnR7QE2itvyrMxCDSEVnMhKdCNb23L2TptUmpvLcb8wfAMLFsSu2yaOtJysep\noH/mvGqlRv2ti2+E2YA0M7Pf83wyV1XmuEsc9tQ225rprDk2uyshUglkDD2235rf\n0QyONq3Aw3BMrO9ss1qj7vdDhVHVsxHnTVbEgrxEWkq2GkVKh9QReMZ2AKxe40j4\nog+OjKXguOCggCZHJyXKxccwqCaeCztbAgMBAAGgIjAgBgkqhkiG9w0BCQIxExMR\nU05NUCBMYWJvcmF0b3JpZXMwDQYJKoZIhvcNAQEFBQADggEBAAihbwmN9M2bsNNm\n9KfxqiGMqqcGCtzIlpDz/2NVwY93cEZsbz3Qscc0QpknRmyTSoDwIG+1nUH0vzkT\nNv8sBmp9I1GdhGg52DIaWwL4t9O5WUHgfHSJpPxZ/zMP2qIsdPJ+8o19BbXRlufc\n73c03H1piGeb9VcePIaulSHI622xukI6f4Sis49vkDaoi+jadbEEb6TYkJQ3AMRD\nWdApGGm0BePdLqboW1Yv70WRRFFD8sxeT7Yw4qrJojdnq0xMHPGfKpf6dJsqWkHk\nb5DRbjil1Zt9pJuF680S9wtBzSi0hsMHXR9TzS7HpMjykL2nmCVY6A78MZapsCzn\nGGbx7DI=\n'

    def setUp(self):
        self.asn1Spec = rfc2314.CertificationRequest()

    def testDerCodec(self):
        substrate = pem.readBase64fromText(self.pem_text)
        asn1Object, rest = der_decoder.decode(substrate, asn1Spec=self.asn1Spec)
        assert not rest
        assert asn1Object.prettyPrint()
        assert der_encoder.encode(asn1Object) == substrate