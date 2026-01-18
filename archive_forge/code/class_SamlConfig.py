from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SamlConfig(_messages.Message):
    """Configuration for the SAML Auth flow.

  Messages:
    AttributeMappingValue: Optional. The mapping of additional user attributes
      like nickname, birthday and address etc.. `key` is the name of this
      additional attribute. `value` is a string presenting as CEL(common
      expression language, go/cel) used for getting the value from the
      resources. Take nickname as an example, in this case, `key` is
      "attribute.nickname" and `value` is "assertion.nickname".

  Fields:
    attributeMapping: Optional. The mapping of additional user attributes like
      nickname, birthday and address etc.. `key` is the name of this
      additional attribute. `value` is a string presenting as CEL(common
      expression language, go/cel) used for getting the value from the
      resources. Take nickname as an example, in this case, `key` is
      "attribute.nickname" and `value` is "assertion.nickname".
    groupPrefix: Optional. Prefix to prepend to group name.
    groupsAttribute: Optional. The SAML attribute to read groups from. This
      value is expected to be a string and will be passed along as-is (with
      the option of being prefixed by the `group_prefix`).
    identityProviderCertificates: Required. The list of IdP certificates to
      validate the SAML response against.
    identityProviderId: Required. The entity ID of the SAML IdP.
    identityProviderSsoUri: Required. The URI where the SAML IdP exposes the
      SSO service.
    userAttribute: Optional. The SAML attribute to read username from. If
      unspecified, the username will be read from the NameID element of the
      assertion in SAML response. This value is expected to be a string and
      will be passed along as-is (with the option of being prefixed by the
      `user_prefix`).
    userPrefix: Optional. Prefix to prepend to user name.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributeMappingValue(_messages.Message):
        """Optional. The mapping of additional user attributes like nickname,
    birthday and address etc.. `key` is the name of this additional attribute.
    `value` is a string presenting as CEL(common expression language, go/cel)
    used for getting the value from the resources. Take nickname as an
    example, in this case, `key` is "attribute.nickname" and `value` is
    "assertion.nickname".

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
    attributeMapping = _messages.MessageField('AttributeMappingValue', 1)
    groupPrefix = _messages.StringField(2)
    groupsAttribute = _messages.StringField(3)
    identityProviderCertificates = _messages.StringField(4, repeated=True)
    identityProviderId = _messages.StringField(5)
    identityProviderSsoUri = _messages.StringField(6)
    userAttribute = _messages.StringField(7)
    userPrefix = _messages.StringField(8)