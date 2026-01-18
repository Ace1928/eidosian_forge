from tests.unit import unittest
from boto.glacier.layer1 import Layer1
class TestGlacierLayer1(unittest.TestCase):
    glacier = True

    def delete_vault(self, vault_name):
        pass

    def test_initialiate_multipart_upload(self):
        glacier = Layer1()
        glacier.create_vault('l1testvault')
        self.addCleanup(glacier.delete_vault, 'l1testvault')
        upload_id = glacier.initiate_multipart_upload('l1testvault', 4 * 1024 * 1024, 'double  spaces  here')['UploadId']
        self.addCleanup(glacier.abort_multipart_upload, 'l1testvault', upload_id)
        response = glacier.list_multipart_uploads('l1testvault')['UploadsList']
        self.assertEqual(len(response), 1)
        self.assertEqual(response[0]['MultipartUploadId'], upload_id)