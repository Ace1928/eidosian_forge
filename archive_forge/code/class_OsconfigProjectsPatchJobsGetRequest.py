from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OsconfigProjectsPatchJobsGetRequest(_messages.Message):
    """A OsconfigProjectsPatchJobsGetRequest object.

  Fields:
    name: Required. Name of the patch in the form `projects/*/patchJobs/*`
  """
    name = _messages.StringField(1, required=True)