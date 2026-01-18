from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsLbObservabilityExtensionsGetRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsLbObservabilityExtensionsGetRequest
  object.

  Fields:
    name: Required. A name of the `LbObservabilityExtension` resource to get.
      Must be in the format `projects/{project}/locations/{location}/lbObserva
      bilityExtensions/{lb_observability_extension}`.
  """
    name = _messages.StringField(1, required=True)