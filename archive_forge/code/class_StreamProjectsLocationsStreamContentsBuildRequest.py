from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamProjectsLocationsStreamContentsBuildRequest(_messages.Message):
    """A StreamProjectsLocationsStreamContentsBuildRequest object.

  Fields:
    buildStreamContentRequest: A BuildStreamContentRequest resource to be
      passed as the request body.
    name: Required. Canonical resource name of the content.
  """
    buildStreamContentRequest = _messages.MessageField('BuildStreamContentRequest', 1)
    name = _messages.StringField(2, required=True)