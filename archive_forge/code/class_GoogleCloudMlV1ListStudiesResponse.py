from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ListStudiesResponse(_messages.Message):
    """A GoogleCloudMlV1ListStudiesResponse object.

  Fields:
    studies: The studies associated with the project.
  """
    studies = _messages.MessageField('GoogleCloudMlV1Study', 1, repeated=True)