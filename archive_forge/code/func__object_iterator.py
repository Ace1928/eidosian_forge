from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import wildcard_iterator
from googlecloudsdk.command_lib.storage.resources import full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import gsutil_full_resource_formatter
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
def _object_iterator(url, fetch_encrypted_object_hashes, halt_on_empty_response, next_page_token, object_state):
    """Iterates through resources matching URL and filter out non-objects."""
    for resource in wildcard_iterator.CloudWildcardIterator(url, error_on_missing_key=False, fetch_encrypted_object_hashes=fetch_encrypted_object_hashes, fields_scope=cloud_api.FieldsScope.FULL, halt_on_empty_response=halt_on_empty_response, next_page_token=next_page_token, object_state=object_state):
        if isinstance(resource, resource_reference.ObjectResource):
            yield resource