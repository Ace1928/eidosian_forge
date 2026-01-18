from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import pkgutil
import tempfile
import textwrap
import six
import boto
from boto import config
import boto.auth
from boto.exception import NoAuthHandlerFound
from boto.gs.connection import GSConnection
from boto.provider import Provider
from boto.pyami.config import BotoConfigLocations
import gslib
from gslib import context_config
from gslib.exception import CommandException
from gslib.utils import system_util
from gslib.utils.constants import DEFAULT_GCS_JSON_API_VERSION
from gslib.utils.constants import DEFAULT_GSUTIL_STATE_DIR
from gslib.utils.constants import SSL_TIMEOUT_SEC
from gslib.utils.constants import UTF8
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import ONE_MIB
import httplib2
from oauth2client.client import HAS_CRYPTO
def _PatchedShouldRetryMethod(self, response, chunked_transfer=False):
    """Replaces boto.s3.key's should_retry() to handle KMS-encrypted objects."""
    provider = self.bucket.connection.provider
    if not chunked_transfer:
        if response.status in [500, 503]:
            return True
        if response.getheader('location'):
            return True
    if 200 <= response.status <= 299:
        self.etag = response.getheader('etag')
        md5 = self.md5
        if isinstance(md5, bytes):
            md5 = md5.decode(UTF8)
        amz_server_side_encryption_customer_algorithm = response.getheader('x-amz-server-side-encryption-customer-algorithm', None)
        goog_customer_managed_encryption = response.getheader('x-goog-encryption-kms-key-name', None)
        if amz_server_side_encryption_customer_algorithm is None and goog_customer_managed_encryption is None:
            if self.etag != '"%s"' % md5:
                raise provider.storage_data_error('ETag from S3 did not match computed MD5. %s vs. %s' % (self.etag, self.md5))
        return True
    if response.status == 400:
        body = response.read()
        err = provider.storage_response_error(response.status, response.reason, body)
        if err.error_code in ['RequestTimeout']:
            raise boto.exception.PleaseRetryException('Saw %s, retrying' % err.error_code, response=response)
    return False