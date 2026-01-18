from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsServicesDeidentifyDeidentifyDicomInstanceRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsServicesDeidentifyDeidentifyDicomInstanceRequest
  object.

  Fields:
    gcsConfigUri: Cloud Storage location to read the JSON DeidentifyConfig
      from, overriding the default config. Must be of the form
      `gs://{bucket_id}/{object_id}`. The Cloud Storage location must grant
      the Cloud IAM role `roles/storage.objectViewer` to the project's Cloud
      Healthcare Service Agent service account.
    httpBody: A HttpBody resource to be passed as the request body.
    name: Required. The name of the service that should handle the request, of
      the form:
      `projects/{project_id}/locations/{location_id}/services/deidentify`.
  """
    gcsConfigUri = _messages.StringField(1)
    httpBody = _messages.MessageField('HttpBody', 2)
    name = _messages.StringField(3, required=True)