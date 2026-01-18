from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ModelGardenSource(_messages.Message):
    """Contains information about the source of the models generated from Model
  Garden.

  Fields:
    publicModelName: Required. The model garden source model resource name.
  """
    publicModelName = _messages.StringField(1)