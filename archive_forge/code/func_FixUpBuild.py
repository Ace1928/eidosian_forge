from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import gzip
import io
import operator
import os
import tarfile
from apitools.base.py import encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
import six
from six.moves import filter  # pylint: disable=redefined-builtin
def FixUpBuild(build, object_ref):
    """Return a modified Build object with run-time values populated.

  Specifically:
  - `source` is pulled from the given object_ref
  - `timeout` comes from the app/cloud_build_timeout property
  - `logsBucket` uses the bucket from object_ref

  Args:
    build: cloudbuild Build message. The Build to modify. Fields 'timeout',
      'source', and 'logsBucket' will be added and may not be given.
    object_ref: storage_util.ObjectReference, the Cloud Storage location of the
      source tarball.

  Returns:
    Build, (copy) of the given Build message with the specified fields
      populated.

  Raises:
    InvalidBuildError: if the Build message had one of the fields this function
      sets pre-populated
  """
    messages = cloudbuild_util.GetMessagesModule()
    build = encoding.CopyProtoMessage(build)
    if build.substitutions:
        build.substitutions.additionalProperties.sort(key=operator.attrgetter('key'))
    _ValidateBuildFields(build, ('source', 'timeout', 'logsBucket'))
    build.timeout = GetServiceTimeoutString(properties.VALUES.app.cloud_build_timeout.Get())
    build.logsBucket = object_ref.bucket
    build.source = messages.Source(storageSource=messages.StorageSource(bucket=object_ref.bucket, object=object_ref.name))
    return build