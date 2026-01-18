import os
import re
import mock
from OpenSSL import crypto
import pytest  # type: ignore
from google.auth import exceptions
from google.auth.transport import _mtls_helper
class TestReadMetadataFile(object):

    def test_success(self):
        metadata_path = os.path.join(pytest.data_dir, 'context_aware_metadata.json')
        metadata = _mtls_helper._read_dca_metadata_file(metadata_path)
        assert 'cert_provider_command' in metadata

    def test_file_not_json(self):
        metadata_path = os.path.join(pytest.data_dir, 'privatekey.pem')
        with pytest.raises(exceptions.ClientCertError):
            _mtls_helper._read_dca_metadata_file(metadata_path)