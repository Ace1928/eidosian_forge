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
def GetServiceTimeoutSeconds(timeout_property_str):
    """Returns the service timeout in seconds given the duration string."""
    if timeout_property_str is None:
        return None
    build_timeout_duration = times.ParseDuration(timeout_property_str, default_suffix='s')
    return int(build_timeout_duration.total_seconds)