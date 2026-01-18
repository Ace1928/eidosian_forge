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
def _SetDefaultLogsBucketBehavior(build_config, messages, arg_bucket_behavior=None):
    """Sets the behavior of the default logs bucket on Build options.

  Args:
    build_config: apitools.base.protorpclite.messages.Message, The Build message
      to analyze.
    messages: API messages class. The CloudBuild API messages.
    arg_bucket_behavior: The default buckets behavior flag.

  Returns:
    build_config: apitools.base.protorpclite.messages.Message, The Build message
      to analyze.
  """
    if arg_bucket_behavior is not None:
        bucket_behavior = flags.GetDefaultBuckestBehavior(arg_bucket_behavior)
        if not build_config.options:
            build_config.options = messages.BuildOptions()
        build_config.options.defaultLogsBucketBehavior = bucket_behavior
    return build_config