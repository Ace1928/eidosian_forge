from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import re
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.core import log
def _get_md5_hash_from_etag(etag, object_url):
    """Returns base64 encoded MD5 hash, if etag is valid MD5."""
    if etag and MD5_REGEX.match(etag):
        encoded_bytes = base64.b64encode(binascii.unhexlify(etag))
        return encoded_bytes.decode(encoding='utf-8')
    else:
        log.debug('Non-MD5 etag ("%s") present for object: %s. Data integrity checks are not possible.', etag, object_url)
    return None