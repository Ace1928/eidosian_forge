from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config
from googlecloudsdk.api_lib.cloudbuild import logs as cb_logs
from googlecloudsdk.api_lib.compute import utils as compute_utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags
from googlecloudsdk.command_lib.builds import staging_bucket_util
from googlecloudsdk.command_lib.cloudbuild import execution
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import times
import six
def _GetBuildTag(builder) -> str:
    """Get the builder tag for input builder useful to cloudbuild.

  Args:
    builder: Google owned builder that needs to be tagged. Any other builders
      are marked as other
  Returns:
    Tag identifying the builder being used.
  """
    if builder == 'gcr.io/buildpacks/builder:latest' or builder == 'gcr.io/buildpacks/builder':
        return 'latest'
    elif builder == 'gcr.io/buildpacks/builder:google-22':
        return 'google22'
    elif builder == 'gcr.io/buildpacks/builder:v1':
        return 'v1'
    elif builder is None:
        return 'default'
    else:
        return 'other'