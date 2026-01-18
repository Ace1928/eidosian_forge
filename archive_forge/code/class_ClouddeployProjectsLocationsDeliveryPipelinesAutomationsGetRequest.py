from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClouddeployProjectsLocationsDeliveryPipelinesAutomationsGetRequest(_messages.Message):
    """A ClouddeployProjectsLocationsDeliveryPipelinesAutomationsGetRequest
  object.

  Fields:
    name: Required. Name of the `Automation`. Format must be `projects/{projec
      t_id}/locations/{location_name}/deliveryPipelines/{pipeline_name}/automa
      tions/{automation_name}`.
  """
    name = _messages.StringField(1, required=True)