from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDataLabelingJobsDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsDataLabelingJobsDeleteRequest object.

  Fields:
    name: Required. The name of the DataLabelingJob to be deleted. Format: `pr
      ojects/{project}/locations/{location}/dataLabelingJobs/{data_labeling_jo
      b}`
  """
    name = _messages.StringField(1, required=True)