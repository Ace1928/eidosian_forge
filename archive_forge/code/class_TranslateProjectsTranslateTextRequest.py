from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsTranslateTextRequest(_messages.Message):
    """A TranslateProjectsTranslateTextRequest object.

  Fields:
    parent: Required. Project or location to make a call. Must refer to a
      caller's project. Format: `projects/{project-number-or-id}` or
      `projects/{project-number-or-id}/locations/{location-id}`. For global
      calls, use `projects/{project-number-or-id}/locations/global` or
      `projects/{project-number-or-id}`. Non-global location is required for
      requests using AutoML models or custom glossaries. Models and glossaries
      must be within the same region (have same location-id), otherwise an
      INVALID_ARGUMENT (400) error is returned.
    translateTextRequest: A TranslateTextRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    translateTextRequest = _messages.MessageField('TranslateTextRequest', 2)