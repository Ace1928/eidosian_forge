from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ImportDataRequest(_messages.Message):
    """Request message for DatasetService.ImportData.

  Fields:
    importConfigs: Required. The desired input locations. The contents of all
      input locations will be imported in one batch.
  """
    importConfigs = _messages.MessageField('GoogleCloudAiplatformV1beta1ImportDataConfig', 1, repeated=True)