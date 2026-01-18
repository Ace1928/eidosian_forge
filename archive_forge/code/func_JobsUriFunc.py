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
def JobsUriFunc(resource):
    """Transform a job resource into a URL string.

  Args:
    resource: The DisplayInfo job object

  Returns:
    URL to the job
  """
    ref = resources.REGISTRY.Parse(resource.id, params={'projectId': properties.VALUES.core.project.GetOrFail, 'location': resource.location}, collection=JOBS_COLLECTION)
    return ref.SelfLink()