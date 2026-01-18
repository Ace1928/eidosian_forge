from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as cloud_errors
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.core import log
def _get_encryption_request_params(gapic_client, decryption_key):
    if decryption_key is not None and decryption_key.type == encryption_util.KeyType.CSEK:
        return gapic_client.types.CommonObjectRequestParams(encryption_algorithm=encryption_util.ENCRYPTION_ALGORITHM, encryption_key_bytes=base64.b64decode(decryption_key.key.encode('utf-8')), encryption_key_sha256_bytes=base64.b64decode(decryption_key.sha256.encode('utf-8')))
    else:
        return None