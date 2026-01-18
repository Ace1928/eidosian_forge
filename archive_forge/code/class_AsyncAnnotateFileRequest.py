from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AsyncAnnotateFileRequest(_messages.Message):
    """An offline file annotation request.

  Fields:
    features: Required. Requested features.
    imageContext: Additional context that may accompany the image(s) in the
      file.
    inputConfig: Required. Information about the input file.
    outputConfig: Required. The desired output location and metadata (e.g.
      format).
  """
    features = _messages.MessageField('Feature', 1, repeated=True)
    imageContext = _messages.MessageField('ImageContext', 2)
    inputConfig = _messages.MessageField('InputConfig', 3)
    outputConfig = _messages.MessageField('OutputConfig', 4)