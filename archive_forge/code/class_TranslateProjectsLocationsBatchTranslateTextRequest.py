from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsBatchTranslateTextRequest(_messages.Message):
    """A TranslateProjectsLocationsBatchTranslateTextRequest object.

  Fields:
    batchTranslateTextRequest: A BatchTranslateTextRequest resource to be
      passed as the request body.
    parent: Required. Location to make a call. Must refer to a caller's
      project. Format: `projects/{project-number-or-id}/locations/{location-
      id}`. The `global` location is not supported for batch translation. Only
      AutoML Translation models or glossaries within the same region (have the
      same location-id) can be used, otherwise an INVALID_ARGUMENT (400) error
      is returned.
  """
    batchTranslateTextRequest = _messages.MessageField('BatchTranslateTextRequest', 1)
    parent = _messages.StringField(2, required=True)