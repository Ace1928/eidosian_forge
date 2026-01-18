from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentEnvironmentsDeleteRequest(_messages.Message):
    """A DialogflowProjectsAgentEnvironmentsDeleteRequest object.

  Fields:
    name: Required. The name of the environment to delete. / Format: -
      `projects//agent/environments/` -
      `projects//locations//agent/environments/` The environment ID for the
      default environment is `-`.
  """
    name = _messages.StringField(1, required=True)