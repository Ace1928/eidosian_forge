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
def GetPipeline(pipeline_name):
    """Gets the delivery pipeline and returns the value of its suspended field.

  Args:
    pipeline_name: str, the canonical resource name of the delivery pipeline

  Returns:
    The pipeline object
  Raises:
    apitools.base.py.HttpError
  """
    try:
        pipeline_obj = delivery_pipeline.DeliveryPipelinesClient().Get(pipeline_name)
        return pipeline_obj
    except apitools_exceptions.HttpError as error:
        log.debug('Failed to get pipeline {}: {}'.format(pipeline_name, error.content))
        log.status.Print('Unable to get delivery pipeline {}'.format(pipeline_name))
        raise error