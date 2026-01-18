from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BigquerydatatransferProjectsLocationsTransferConfigsRunsGetRequest(_messages.Message):
    """A BigquerydatatransferProjectsLocationsTransferConfigsRunsGetRequest
  object.

  Fields:
    name: Required. The field will contain name of the resource requested, for
      example:
      `projects/{project_id}/transferConfigs/{config_id}/runs/{run_id}` or `pr
      ojects/{project_id}/locations/{location_id}/transferConfigs/{config_id}/
      runs/{run_id}`
  """
    name = _messages.StringField(1, required=True)