from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RunNamespacesServicesGetRequest(_messages.Message):
    """A RunNamespacesServicesGetRequest object.

  Fields:
    name: Required. The fully qualified name of the service to retrieve. It
      can be any of the following forms: *
      `namespaces/{project_id_or_number}/services/{service_name}` (only when
      the `endpoint` is regional) * `projects/{project_id_or_number}/locations
      /{region}/services/{service_name}` * `projects/{project_id_or_number}/re
      gions/{region}/services/{service_name}`
  """
    name = _messages.StringField(1, required=True)