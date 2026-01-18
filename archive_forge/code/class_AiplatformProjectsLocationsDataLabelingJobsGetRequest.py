from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDataLabelingJobsGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsDataLabelingJobsGetRequest object.

  Fields:
    name: Required. The name of the DataLabelingJob. Format: `projects/{projec
      t}/locations/{location}/dataLabelingJobs/{data_labeling_job}`
  """
    name = _messages.StringField(1, required=True)