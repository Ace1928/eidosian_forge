from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.app import util
from googlecloudsdk.api_lib.tasks import task_queues_convertors as convertors
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.command_lib.tasks import flags
from googlecloudsdk.command_lib.tasks import parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
import six
from six.moves import urllib
def _CreateUniqueJobKeyForYamlJob(job):
    """Creates a key from the YAML job instance's attributes passed as input.

  Args:
    job: An instance of a parsed YAML job object.
  Returns:
    A tuple of attributes used as a key to identify this job.
  """
    retry_params = job.retry_parameters
    return (job.schedule, job.timezone if job.timezone else 'UTC', job.url, job.description, retry_params.min_backoff_seconds if retry_params else None, retry_params.max_backoff_seconds if retry_params else None, retry_params.max_doublings if retry_params else None, convertors.CheckAndConvertStringToFloatIfApplicable(retry_params.job_age_limit) if retry_params else None, retry_params.job_retry_limit if retry_params else None, job.target)