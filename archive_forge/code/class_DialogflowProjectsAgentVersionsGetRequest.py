from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsAgentVersionsGetRequest(_messages.Message):
    """A DialogflowProjectsAgentVersionsGetRequest object.

  Fields:
    name: Required. The name of the version. Supported formats: -
      `projects//agent/versions/` - `projects//locations//agent/versions/`
  """
    name = _messages.StringField(1, required=True)