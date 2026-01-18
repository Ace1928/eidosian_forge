from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import sys
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.dataproc import exceptions as dp_exceptions
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six.moves.urllib.parse
def GetObjectRef(path, messages):
    """Build an Object proto message from a GCS path."""
    resource = resources.REGISTRY.ParseStorageURL(path)
    return messages.Object(bucket=resource.bucket, name=resource.object)