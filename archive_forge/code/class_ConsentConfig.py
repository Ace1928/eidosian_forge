from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsentConfig(_messages.Message):
    """Configures whether to enforce consent for the FHIR store and which
  consent enforcement version is being used.

  Enums:
    VersionValueValuesEnum: Required. Specifies which consent enforcement
      version is being used for this FHIR store. This field can only be set
      once by either CreateFhirStore or UpdateFhirStore. After that, you must
      call ApplyConsents to change the version.

  Fields:
    accessDeterminationLogConfig: Optional. Specifies how the server logs the
      consent-aware requests. If not specified, the
      `AccessDeterminationLogConfig.LogLevel.MINIMUM` option is used.
    accessEnforced: Optional. If set to true, when accessing FHIR resources,
      the consent headers will be verified against consents given by patients.
      See the ConsentEnforcementVersion for the supported consent headers.
    consentHeaderHandling: Optional. Different options to configure the
      behaviour of the server when handling the `X-Consent-Scope` header.
    enforcedAdminConsents: The versioned names of the enforced admin Consent
      resource(s), in the format `projects/{project_id}/locations/{location}/d
      atasets/{dataset_id}/fhirStores/{fhir_store_id}/fhir/Consent/{resource_i
      d}/_history/{version_id}`. For FHIR stores with
      `disable_resource_versioning=true`, the format is `projects/{project_id}
      /locations/{location}/datasets/{dataset_id}/fhirStores/{fhir_store_id}/f
      hir/Consent/{resource_id}`. This field can only be updated using
      ApplyAdminConsents.
    version: Required. Specifies which consent enforcement version is being
      used for this FHIR store. This field can only be set once by either
      CreateFhirStore or UpdateFhirStore. After that, you must call
      ApplyConsents to change the version.
  """

    class VersionValueValuesEnum(_messages.Enum):
        """Required. Specifies which consent enforcement version is being used
    for this FHIR store. This field can only be set once by either
    CreateFhirStore or UpdateFhirStore. After that, you must call
    ApplyConsents to change the version.

    Values:
      CONSENT_ENFORCEMENT_VERSION_UNSPECIFIED: Users must specify an
        enforcement version or an error is returned.
      V1: Enforcement version 1. See the [FHIR Consent resources in the Cloud
        Healthcare API](https://cloud.google.com/healthcare-api/docs/fhir-
        consent) guide for more details.
    """
        CONSENT_ENFORCEMENT_VERSION_UNSPECIFIED = 0
        V1 = 1
    accessDeterminationLogConfig = _messages.MessageField('AccessDeterminationLogConfig', 1)
    accessEnforced = _messages.BooleanField(2)
    consentHeaderHandling = _messages.MessageField('ConsentHeaderHandling', 3)
    enforcedAdminConsents = _messages.StringField(4, repeated=True)
    version = _messages.EnumField('VersionValueValuesEnum', 5)