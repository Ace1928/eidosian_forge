from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class VersionAliasesValue(_messages.Message):
    """Optional. Mapping from version alias to version name. A version alias
    is a string with a maximum length of 63 characters and can contain
    uppercase and lowercase letters, numerals, and the hyphen (`-`) and
    underscore ('_') characters. An alias string must start with a letter and
    cannot be the string 'latest' or 'NEW'. No more than 50 aliases can be
    assigned to a given secret. Version-Alias pairs will be viewable via
    GetSecret and modifiable via UpdateSecret. Access by alias is only be
    supported on GetSecretVersion and AccessSecretVersion.

    Messages:
      AdditionalProperty: An additional property for a VersionAliasesValue
        object.

    Fields:
      additionalProperties: Additional properties of type VersionAliasesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a VersionAliasesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.IntegerField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)