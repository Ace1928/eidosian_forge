from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplyAdminConsentsRequest(_messages.Message):
    """Request to apply the admin Consent resources for the specified FHIR
  store.

  Fields:
    newConsentsList: A new list of admin Consent resources to be applied. Any
      existing enforced Consents, which are specified in
      `consent_config.enforced_admin_consents` of the FhirStore, that are not
      part of this list will be disabled. An empty list is equivalent to
      clearing or disabling all Consents enforced on the FHIR store. When a
      FHIR store has `disable_resource_versioning=true` and this list contains
      a Consent resource that exists in
      `consent_config.enforced_admin_consents`, the method enforces any
      updates to the existing resource since the last enforcement. If the
      existing resource hasn't been updated since the last enforcement, the
      resource is unaffected. After the method finishes, the resulting consent
      enforcement model is determined by the contents of the Consent
      resource(s) when the method was called: * When
      `disable_resource_versioning=true`, the result is identical to the
      current resource(s) in the FHIR store. * When
      `disable_resource_versioning=false`, the result is based on the
      historical version(s) of the Consent resource(s) at the point in time
      when the method was called. At most 200 Consents can be specified.
    validateOnly: If true, the method only validates Consent resources to make
      sure they are supported. Otherwise, the method applies the aggregate
      consent information to update the enforcement model and reindex the FHIR
      resources. If all Consent resources can be applied successfully, the
      ApplyAdminConsentsResponse is returned containing the following fields:
      * `consent_apply_success` to indicate the number of Consent resources
      applied. * `affected_resources` to indicate the number of resources that
      might have had their consent access changed. If, however, one or more
      Consent resources are unsupported or cannot be applied, the method fails
      and ApplyAdminConsentsErrorDetail is is returned with details about the
      unsupported Consent resources.
  """
    newConsentsList = _messages.MessageField('AdminConsents', 1)
    validateOnly = _messages.BooleanField(2)