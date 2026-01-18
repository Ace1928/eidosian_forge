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
def GetMaxConcurrentCompressedUploads():
    """Gets the max concurrent transport compressed uploads allowed in parallel.

  Returns:
    The max number of concurrent transport compressed uploads allowed in
    parallel without exceeding the max_upload_compression_buffer_size.
  """
    upload_chunk_size = GetJsonResumableChunkSize()
    compression_chunk_size = 16 * ONE_MIB
    total_upload_size = upload_chunk_size + compression_chunk_size + 17 + 5 * ((compression_chunk_size - 1) / 16383 + 1)
    max_concurrent_uploads = GetMaxUploadCompressionBufferSize() / total_upload_size
    if max_concurrent_uploads <= 0:
        max_concurrent_uploads = 1
    return max_concurrent_uploads