from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsModelsCreateRequest(_messages.Message):
    """A TranslateProjectsLocationsModelsCreateRequest object.

  Fields:
    model: A Model resource to be passed as the request body.
    parent: Required. The project name, in form of
      `projects/{project}/locations/{location}`
  """
    model = _messages.MessageField('Model', 1)
    parent = _messages.StringField(2, required=True)