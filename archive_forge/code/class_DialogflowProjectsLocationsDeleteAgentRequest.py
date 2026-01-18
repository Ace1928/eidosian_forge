from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsDeleteAgentRequest(_messages.Message):
    """A DialogflowProjectsLocationsDeleteAgentRequest object.

  Fields:
    parent: Required. The project that the agent to delete is associated with.
      Format: `projects/`.
  """
    parent = _messages.StringField(1, required=True)