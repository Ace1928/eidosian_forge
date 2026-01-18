from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentEntityTypesDeleteRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentEntityTypesDeleteRequest object.

  Fields:
    name: Required. The name of the entity type to delete. Format:
      `projects//agent/entityTypes/`.
  """
    name = _messages.StringField(1, required=True)