from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p3beta1AnnotateFileResponse(_messages.Message):
    """Response to a single file annotation request. A file may contain one or
  more images, which individually have their own responses.

  Fields:
    error: If set, represents the error message for the failed request. The
      `responses` field will not be set in this case.
    inputConfig: Information about the file for which this response is
      generated.
    responses: Individual responses to images found within the file. This
      field will be empty if the `error` field is set.
    totalPages: This field gives the total number of pages in the file.
  """
    error = _messages.MessageField('Status', 1)
    inputConfig = _messages.MessageField('GoogleCloudVisionV1p3beta1InputConfig', 2)
    responses = _messages.MessageField('GoogleCloudVisionV1p3beta1AnnotateImageResponse', 3, repeated=True)
    totalPages = _messages.IntegerField(4, variant=_messages.Variant.INT32)