from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsFeaturestoresGetRequest(_messages.Message):
    """A AiplatformProjectsLocationsFeaturestoresGetRequest object.

  Fields:
    name: Required. The name of the Featurestore resource.
  """
    name = _messages.StringField(1, required=True)