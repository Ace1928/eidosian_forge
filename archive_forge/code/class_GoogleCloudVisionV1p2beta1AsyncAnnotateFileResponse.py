from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p2beta1AsyncAnnotateFileResponse(_messages.Message):
    """The response for a single offline file annotation request.

  Fields:
    outputConfig: The output location and metadata from
      AsyncAnnotateFileRequest.
  """
    outputConfig = _messages.MessageField('GoogleCloudVisionV1p2beta1OutputConfig', 1)