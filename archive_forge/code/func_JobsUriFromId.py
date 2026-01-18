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
def JobsUriFromId(job_id, region_id):
    """Transform a job ID into a URL string.

  Args:
    job_id: The job ID
    region_id: Region ID of the job's regional endpoint.

  Returns:
    URL to the job
  """
    ref = resources.REGISTRY.Parse(job_id, params={'projectId': properties.VALUES.core.project.GetOrFail, 'location': region_id}, collection=JOBS_COLLECTION)
    return ref.SelfLink()