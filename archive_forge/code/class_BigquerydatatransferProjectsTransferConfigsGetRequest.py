from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsTransferConfigsGetRequest(_messages.Message):
    """A BigquerydatatransferProjectsTransferConfigsGetRequest object.

  Fields:
    name: Required. The field will contain name of the resource requested, for
      example: `projects/{project_id}/transferConfigs/{config_id}` or `project
      s/{project_id}/locations/{location_id}/transferConfigs/{config_id}`
  """
    name = _messages.StringField(1, required=True)