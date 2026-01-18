from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApphubProjectsLocationsApplicationsServicesGetRequest(_messages.Message):
    """A ApphubProjectsLocationsApplicationsServicesGetRequest object.

  Fields:
    name: Required. Fully qualified name of the Service to fetch. Expected
      format: `projects/{project}/locations/{location}/applications/{applicati
      on}/services/{service}`.
  """
    name = _messages.StringField(1, required=True)