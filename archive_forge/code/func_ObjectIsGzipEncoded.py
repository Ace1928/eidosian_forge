from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def ObjectIsGzipEncoded(obj_metadata):
    """Returns true if the apitools_messages.Object has gzip content-encoding."""
    return obj_metadata.contentEncoding and obj_metadata.contentEncoding.lower().endswith('gzip')