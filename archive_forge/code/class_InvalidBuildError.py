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
class InvalidBuildError(ValueError):
    """Error indicating that ExecuteCloudBuild was given a bad Build message."""

    def __init__(self, field):
        super(InvalidBuildError, self).__init__('Field [{}] was provided, but should not have been. You may be using an improper Cloud Build pipeline.'.format(field))