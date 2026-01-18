from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import uuid
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.clouddeploy import client_util
from googlecloudsdk.api_lib.clouddeploy import delivery_pipeline
from googlecloudsdk.command_lib.deploy import exceptions as cd_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ThrowIfPipelineSuspended(pipeline_obj, suspended_pipeline_msg):
    """Checks if the delivery pipeline associated with the release is suspended.

  Args:
    pipeline_obj: protorpc.messages.Message, delivery pipeline object.
    suspended_pipeline_msg: str, error msg to show the user if the pipeline is
      suspended.

  Raises:
    googlecloudsdk.command_lib.deploy.PipelineSuspendedError if the pipeline is
    suspended
  """
    if pipeline_obj.suspended:
        raise cd_exceptions.PipelineSuspendedError(pipeline_name=pipeline_obj.name, failed_activity_msg=suspended_pipeline_msg)