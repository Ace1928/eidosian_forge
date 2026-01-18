from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import fast_crc32c_util
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.util import crc32c
from googlecloudsdk.core import log
def _get_resource_iterator(url_strings):
    """Wildcard matches and recurses into top-level of buckets."""
    any_url_matched = False
    for url_string in url_strings:
        wildcard_expanded_iterator = wildcard_iterator.get_wildcard_iterator(url_string, error_on_missing_key=False, fetch_encrypted_object_hashes=True)
        this_url_matched = False
        for wildcard_expanded_resource in wildcard_expanded_iterator:
            if _is_object_or_file_resource(wildcard_expanded_resource):
                any_url_matched = this_url_matched = True
                yield wildcard_expanded_resource
            elif isinstance(wildcard_expanded_resource.storage_url, storage_url.CloudUrl) and wildcard_expanded_resource.storage_url.is_bucket():
                bucket_expanded_iterator = wildcard_iterator.get_wildcard_iterator(wildcard_expanded_resource.storage_url.join('*').url_string, error_on_missing_key=False)
                for bucket_expanded_resource in bucket_expanded_iterator:
                    if isinstance(bucket_expanded_resource, resource_reference.ObjectResource):
                        any_url_matched = this_url_matched = True
                        yield bucket_expanded_resource
        if not this_url_matched:
            log.warning('No matches found for {}'.format(url_string))
    if not any_url_matched:
        raise errors.InvalidUrlError('No URLS matched.')