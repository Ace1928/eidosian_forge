from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkforcePoolProvider(_messages.Message):
    """A configuration for an external identity provider.

  Enums:
    StateValueValuesEnum: Output only. The state of the provider.

  Messages:
    AttributeMappingValue: Required. Maps attributes from the authentication
      credentials issued by an external identity provider to Google Cloud
      attributes, such as `subject` and `segment`. Each key must be a string
      specifying the Google Cloud IAM attribute to map to. The following keys
      are supported: * `google.subject`: The principal IAM is authenticating.
      You can reference this value in IAM bindings. This is also the subject
      that appears in Cloud Logging logs. This is a required field and the
      mapped subject cannot exceed 127 bytes. * `google.groups`: Groups the
      authenticating user belongs to. You can grant groups access to resources
      using an IAM `principalSet` binding; access applies to all members of
      the group. * `google.display_name`: The name of the authenticated user.
      This is an optional field and the mapped display name cannot exceed 100
      bytes. If not set, `google.subject` will be displayed instead. This
      attribute cannot be referenced in IAM bindings. *
      `google.profile_photo`: The URL that specifies the authenticated user's
      thumbnail photo. This is an optional field. When set, the image will be
      visible as the user's profile picture. If not set, a generic user icon
      will be displayed instead. This attribute cannot be referenced in IAM
      bindings. * `google.posix_username`: The Linux username used by OS
      Login. This is an optional field and the mapped POSIX username cannot
      exceed 32 characters, The key must match the regex
      "^a-zA-Z0-9._{0,31}$". This attribute cannot be referenced in IAM
      bindings. You can also provide custom attributes by specifying
      `attribute.{custom_attribute}`, where {custom_attribute} is the name of
      the custom attribute to be mapped. You can define a maximum of 50 custom
      attributes. The maximum length of a mapped attribute key is 100
      characters, and the key may only contain the characters [a-z0-9_]. You
      can reference these attributes in IAM policies to define fine-grained
      access for a workforce pool to Google Cloud resources. For example: *
      `google.subject`: `principal://iam.googleapis.com/locations/global/workf
      orcePools/{pool}/subject/{value}` * `google.groups`: `principalSet://iam
      .googleapis.com/locations/global/workforcePools/{pool}/group/{value}` *
      `attribute.{custom_attribute}`: `principalSet://iam.googleapis.com/locat
      ions/global/workforcePools/{pool}/attribute.{custom_attribute}/{value}`
      Each value must be a [Common Expression Language]
      (https://opensource.google/projects/cel) function that maps an identity
      provider credential to the normalized attribute specified by the
      corresponding map key. You can use the `assertion` keyword in the
      expression to access a JSON representation of the authentication
      credential issued by the provider. The maximum length of an attribute
      mapping expression is 2048 characters. When evaluated, the total size of
      all mapped attributes must not exceed 4KB. For OIDC providers, you must
      supply a custom mapping that includes the `google.subject` attribute.
      For example, the following maps the `sub` claim of the incoming
      credential to the `subject` attribute on a Google token: ```
      {"google.subject": "assertion.sub"} ```

  Fields:
    attributeCondition: A [Common Expression
      Language](https://opensource.google/projects/cel) expression, in plain
      text, to restrict what otherwise valid authentication credentials issued
      by the provider should not be accepted. The expression must output a
      boolean representing whether to allow the federation. The following
      keywords may be referenced in the expressions: * `assertion`: JSON
      representing the authentication credential issued by the provider. *
      `google`: The Google attributes mapped from the assertion in the
      `attribute_mappings`. `google.profile_photo`, `google.display_name` and
      `google.posix_username` are not supported. * `attribute`: The custom
      attributes mapped from the assertion in the `attribute_mappings`. The
      maximum length of the attribute condition expression is 4096 characters.
      If unspecified, all valid authentication credentials will be accepted.
      The following example shows how to only allow credentials with a mapped
      `google.groups` value of `admins`: ``` "'admins' in google.groups" ```
    attributeMapping: Required. Maps attributes from the authentication
      credentials issued by an external identity provider to Google Cloud
      attributes, such as `subject` and `segment`. Each key must be a string
      specifying the Google Cloud IAM attribute to map to. The following keys
      are supported: * `google.subject`: The principal IAM is authenticating.
      You can reference this value in IAM bindings. This is also the subject
      that appears in Cloud Logging logs. This is a required field and the
      mapped subject cannot exceed 127 bytes. * `google.groups`: Groups the
      authenticating user belongs to. You can grant groups access to resources
      using an IAM `principalSet` binding; access applies to all members of
      the group. * `google.display_name`: The name of the authenticated user.
      This is an optional field and the mapped display name cannot exceed 100
      bytes. If not set, `google.subject` will be displayed instead. This
      attribute cannot be referenced in IAM bindings. *
      `google.profile_photo`: The URL that specifies the authenticated user's
      thumbnail photo. This is an optional field. When set, the image will be
      visible as the user's profile picture. If not set, a generic user icon
      will be displayed instead. This attribute cannot be referenced in IAM
      bindings. * `google.posix_username`: The Linux username used by OS
      Login. This is an optional field and the mapped POSIX username cannot
      exceed 32 characters, The key must match the regex
      "^a-zA-Z0-9._{0,31}$". This attribute cannot be referenced in IAM
      bindings. You can also provide custom attributes by specifying
      `attribute.{custom_attribute}`, where {custom_attribute} is the name of
      the custom attribute to be mapped. You can define a maximum of 50 custom
      attributes. The maximum length of a mapped attribute key is 100
      characters, and the key may only contain the characters [a-z0-9_]. You
      can reference these attributes in IAM policies to define fine-grained
      access for a workforce pool to Google Cloud resources. For example: *
      `google.subject`: `principal://iam.googleapis.com/locations/global/workf
      orcePools/{pool}/subject/{value}` * `google.groups`: `principalSet://iam
      .googleapis.com/locations/global/workforcePools/{pool}/group/{value}` *
      `attribute.{custom_attribute}`: `principalSet://iam.googleapis.com/locat
      ions/global/workforcePools/{pool}/attribute.{custom_attribute}/{value}`
      Each value must be a [Common Expression Language]
      (https://opensource.google/projects/cel) function that maps an identity
      provider credential to the normalized attribute specified by the
      corresponding map key. You can use the `assertion` keyword in the
      expression to access a JSON representation of the authentication
      credential issued by the provider. The maximum length of an attribute
      mapping expression is 2048 characters. When evaluated, the total size of
      all mapped attributes must not exceed 4KB. For OIDC providers, you must
      supply a custom mapping that includes the `google.subject` attribute.
      For example, the following maps the `sub` claim of the incoming
      credential to the `subject` attribute on a Google token: ```
      {"google.subject": "assertion.sub"} ```
    description: A user-specified description of the provider. Cannot exceed
      256 characters.
    disabled: Disables the workforce pool provider. You cannot use a disabled
      provider to exchange tokens. However, existing tokens still grant
      access.
    displayName: A user-specified display name for the provider. Cannot exceed
      32 characters.
    expireTime: Output only. Time after which the workload pool provider will
      be permanently purged and cannot be recovered.
    extraAttributesOauth2Client: Optional. The configuration for OAuth 2.0
      client used to get the additional user attributes. This should be used
      when users can't get the desired claims in authentication credentials.
      Currently this configuration is only supported with OIDC protocol.
    name: Output only. The resource name of the provider. Format: `locations/{
      location}/workforcePools/{workforce_pool_id}/providers/{provider_id}`
    oidc: An OpenId Connect 1.0 identity provider configuration.
    saml: A SAML identity provider configuration.
    state: Output only. The state of the provider.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the provider.

    Values:
      STATE_UNSPECIFIED: State unspecified.
      ACTIVE: The provider is active and may be used to validate
        authentication credentials.
      DELETED: The provider is soft-deleted. Soft-deleted providers are
        permanently deleted after approximately 30 days. You can restore a
        soft-deleted provider using UndeleteWorkforcePoolProvider.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        DELETED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributeMappingValue(_messages.Message):
        """Required. Maps attributes from the authentication credentials issued
    by an external identity provider to Google Cloud attributes, such as
    `subject` and `segment`. Each key must be a string specifying the Google
    Cloud IAM attribute to map to. The following keys are supported: *
    `google.subject`: The principal IAM is authenticating. You can reference
    this value in IAM bindings. This is also the subject that appears in Cloud
    Logging logs. This is a required field and the mapped subject cannot
    exceed 127 bytes. * `google.groups`: Groups the authenticating user
    belongs to. You can grant groups access to resources using an IAM
    `principalSet` binding; access applies to all members of the group. *
    `google.display_name`: The name of the authenticated user. This is an
    optional field and the mapped display name cannot exceed 100 bytes. If not
    set, `google.subject` will be displayed instead. This attribute cannot be
    referenced in IAM bindings. * `google.profile_photo`: The URL that
    specifies the authenticated user's thumbnail photo. This is an optional
    field. When set, the image will be visible as the user's profile picture.
    If not set, a generic user icon will be displayed instead. This attribute
    cannot be referenced in IAM bindings. * `google.posix_username`: The Linux
    username used by OS Login. This is an optional field and the mapped POSIX
    username cannot exceed 32 characters, The key must match the regex
    "^a-zA-Z0-9._{0,31}$". This attribute cannot be referenced in IAM
    bindings. You can also provide custom attributes by specifying
    `attribute.{custom_attribute}`, where {custom_attribute} is the name of
    the custom attribute to be mapped. You can define a maximum of 50 custom
    attributes. The maximum length of a mapped attribute key is 100
    characters, and the key may only contain the characters [a-z0-9_]. You can
    reference these attributes in IAM policies to define fine-grained access
    for a workforce pool to Google Cloud resources. For example: *
    `google.subject`: `principal://iam.googleapis.com/locations/global/workfor
    cePools/{pool}/subject/{value}` * `google.groups`: `principalSet://iam.goo
    gleapis.com/locations/global/workforcePools/{pool}/group/{value}` *
    `attribute.{custom_attribute}`: `principalSet://iam.googleapis.com/locatio
    ns/global/workforcePools/{pool}/attribute.{custom_attribute}/{value}` Each
    value must be a [Common Expression Language]
    (https://opensource.google/projects/cel) function that maps an identity
    provider credential to the normalized attribute specified by the
    corresponding map key. You can use the `assertion` keyword in the
    expression to access a JSON representation of the authentication
    credential issued by the provider. The maximum length of an attribute
    mapping expression is 2048 characters. When evaluated, the total size of
    all mapped attributes must not exceed 4KB. For OIDC providers, you must
    supply a custom mapping that includes the `google.subject` attribute. For
    example, the following maps the `sub` claim of the incoming credential to
    the `subject` attribute on a Google token: ``` {"google.subject":
    "assertion.sub"} ```

    Messages:
      AdditionalProperty: An additional property for a AttributeMappingValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        AttributeMappingValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributeMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributeCondition = _messages.StringField(1)
    attributeMapping = _messages.MessageField('AttributeMappingValue', 2)
    description = _messages.StringField(3)
    disabled = _messages.BooleanField(4)
    displayName = _messages.StringField(5)
    expireTime = _messages.StringField(6)
    extraAttributesOauth2Client = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderExtraAttributesOAuth2Client', 7)
    name = _messages.StringField(8)
    oidc = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderOidc', 9)
    saml = _messages.MessageField('GoogleIamAdminV1WorkforcePoolProviderSaml', 10)
    state = _messages.EnumField('StateValueValuesEnum', 11)