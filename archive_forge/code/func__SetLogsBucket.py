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
def _SetLogsBucket(build_config, arg_gcs_log_dir):
    """Set a Google Cloud Storage directory to hold build logs."""
    if arg_gcs_log_dir:
        try:
            gcs_log_dir = resources.REGISTRY.Parse(arg_gcs_log_dir, collection='storage.objects')
            build_config.logsBucket = 'gs://' + gcs_log_dir.bucket + '/' + gcs_log_dir.object
            return build_config
        except resources.WrongResourceCollectionException:
            pass
        try:
            gcs_log_dir = resources.REGISTRY.Parse(arg_gcs_log_dir, collection='storage.buckets')
            build_config.logsBucket = 'gs://' + gcs_log_dir.bucket
        except resources.WrongResourceCollectionException as e:
            raise resources.WrongResourceCollectionException(expected='storage.buckets,storage.objects', got=e.got, path=e.path)
    return build_config