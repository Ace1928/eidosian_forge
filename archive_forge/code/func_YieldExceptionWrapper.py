from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.api_lib.dataflow import exceptions as dataflow_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def YieldExceptionWrapper(generator, job_id='', project_id='', region_id=''):
    """Wraps a generator to catch any exceptions.

  Args:
    generator: The error exceptions.HttpError thrown by the API client.
    job_id: The job ID that was used in the command.
    project_id: The project ID that was used in the command.
    region_id: The region ID that was used in the command.

  Yields:
    The generated object.

  Raises:
    dataflow_exceptions.ServiceException: An exception for errors raised by
      the service.
  """
    try:
        for x in generator:
            yield x
    except exceptions.HttpError as e:
        raise dataflow_exceptions.ServiceException(MakeErrorMessage(e, job_id, project_id, region_id))