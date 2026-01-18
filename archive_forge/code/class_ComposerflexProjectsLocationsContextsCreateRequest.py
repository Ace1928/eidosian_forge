from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComposerflexProjectsLocationsContextsCreateRequest(_messages.Message):
    """A ComposerflexProjectsLocationsContextsCreateRequest object.

  Fields:
    context: A Context resource to be passed as the request body.
    parent: Parent resource of the context to create. The parent must be of
      the form "projects/{projectId}/locations/{locationId}".
  """
    context = _messages.MessageField('Context', 1)
    parent = _messages.StringField(2, required=True)