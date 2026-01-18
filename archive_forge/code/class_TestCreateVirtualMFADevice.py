from base64 import b64decode
from boto.compat import json
from boto.iam.connection import IAMConnection
from tests.unit import AWSMockServiceTestCase
class TestCreateVirtualMFADevice(AWSMockServiceTestCase):
    connection_class = IAMConnection

    def default_body(self):
        return b'\n            <CreateVirtualMFADeviceResponse>\n              <CreateVirtualMFADeviceResult>\n                <VirtualMFADevice>\n                  <SerialNumber>arn:aws:iam::123456789012:mfa/ExampleName</SerialNumber>\n                  <Base32StringSeed>2K5K5XTLA7GGE75TQLYEXAMPLEEXAMPLEEXAMPLECHDFW4KJYZ6\n                  UFQ75LL7COCYKM</Base32StringSeed>\n                  <QRCodePNG>89504E470D0A1A0AASDFAHSDFKJKLJFKALSDFJASDF</QRCodePNG>\n                </VirtualMFADevice>\n              </CreateVirtualMFADeviceResult>\n              <ResponseMetadata>\n                <RequestId>7a62c49f-347e-4fc4-9331-6e8eEXAMPLE</RequestId>\n              </ResponseMetadata>\n            </CreateVirtualMFADeviceResponse>\n        '

    def test_create_virtual_mfa_device(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.create_virtual_mfa_device('/', 'ExampleName')
        self.assert_request_parameters({'Path': '/', 'VirtualMFADeviceName': 'ExampleName', 'Action': 'CreateVirtualMFADevice'}, ignore_params_values=['Version'])
        self.assertEquals(response['create_virtual_mfa_device_response']['create_virtual_mfa_device_result']['virtual_mfa_device']['serial_number'], 'arn:aws:iam::123456789012:mfa/ExampleName')