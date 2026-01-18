from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirResourceValidateRequest(_messages.Message):
    """A
  HealthcareProjectsLocationsDatasetsFhirStoresFhirResourceValidateRequest
  object.

  Fields:
    httpBody: A HttpBody resource to be passed as the request body.
    parent: The name of the FHIR store that holds the profiles being used for
      validation.
    profile: The canonical URL of a profile that this resource should be
      validated against. For example, to validate a Patient resource against
      the US Core Patient profile this parameter would be
      `http://hl7.org/fhir/us/core/StructureDefinition/us-core-patient`. A
      StructureDefinition with this canonical URL must exist in the FHIR
      store.
    type: The FHIR resource type of the resource being validated. For a
      complete list, see the FHIR Resource Index ([DSTU2](http://hl7.org/imple
      ment/standards/fhir/DSTU2/resourcelist.html),
      [STU3](http://hl7.org/implement/standards/fhir/STU3/resourcelist.html),
      or [R4](http://hl7.org/implement/standards/fhir/R4/resourcelist.html)).
      Must match the resource type in the provided content.
  """
    httpBody = _messages.MessageField('HttpBody', 1)
    parent = _messages.StringField(2, required=True)
    profile = _messages.StringField(3)
    type = _messages.StringField(4, required=True)