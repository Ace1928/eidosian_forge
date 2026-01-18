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
def _is_object_or_file_resource(resource):
    return isinstance(resource, (resource_reference.ObjectResource, resource_reference.FileObjectResource))