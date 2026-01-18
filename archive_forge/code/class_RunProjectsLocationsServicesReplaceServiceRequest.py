from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunProjectsLocationsServicesReplaceServiceRequest(_messages.Message):
    """A RunProjectsLocationsServicesReplaceServiceRequest object.

  Fields:
    dryRun: Indicates that the server should validate the request and populate
      default values without persisting the request. Supported values: `all`
    name: Required. The fully qualified name of the service to replace. It can
      be any of the following forms: *
      `namespaces/{project_id_or_number}/services/{service_name}` (only when
      the `endpoint` is regional) * `projects/{project_id_or_number}/locations
      /{region}/services/{service_name}` * `projects/{project_id_or_number}/re
      gions/{region}/services/{service_name}`
    service: A Service resource to be passed as the request body.
  """
    dryRun = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    service = _messages.MessageField('Service', 3)