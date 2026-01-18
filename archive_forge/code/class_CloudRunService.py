from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudRunService(_messages.Message):
    """Represents a Cloud Run service destination.

  Fields:
    path: Optional. The relative path on the Cloud Run service the events
      should be sent to. The value must conform to the definition of URI path
      segment (section 3.3 of RFC2396). Examples: "/route", "route",
      "route/subroute".
    region: Required. The region the Cloud Run service is deployed in.
    service: Required. The name of the Cloud run service being addressed. See
      https://cloud.google.com/run/docs/reference/rest/v1/namespaces.services.
      Only services located in the same project of the trigger object can be
      addressed.
  """
    path = _messages.StringField(1)
    region = _messages.StringField(2)
    service = _messages.StringField(3)