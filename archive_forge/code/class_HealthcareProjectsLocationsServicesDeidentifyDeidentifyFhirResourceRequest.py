from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsServicesDeidentifyDeidentifyFhirResourceRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsServicesDeidentifyDeidentifyFhirResourceRequest
  object.

  Enums:
    VersionValueValuesEnum:

  Fields:
    gcsConfigUri: Cloud Storage location to read the JSON DeidentifyConfig
      from.
    httpBody: A HttpBody resource to be passed as the request body.
    name: Required. The name of the service that should handle the request, of
      the form:
      `projects/{project_id}/locations/{location_id}/services/deidentify`.
    version: A VersionValueValuesEnum attribute.
  """

    class VersionValueValuesEnum(_messages.Enum):
        """VersionValueValuesEnum enum type.

    Values:
      VERSION_UNSPECIFIED: VERSION_UNSPECIFIED is treated as STU3.
      DSTU2: FHIR version DSTU2.
      STU3: FHIR version STU3.
      R4: FHIR version R4.
    """
        VERSION_UNSPECIFIED = 0
        DSTU2 = 1
        STU3 = 2
        R4 = 3
    gcsConfigUri = _messages.StringField(1)
    httpBody = _messages.MessageField('HttpBody', 2)
    name = _messages.StringField(3, required=True)
    version = _messages.EnumField('VersionValueValuesEnum', 4)