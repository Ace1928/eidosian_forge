from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleBigtableAdminV2AuthorizedViewSubsetView(_messages.Message):
    """Defines a simple AuthorizedView that is a subset of the underlying
  Table.

  Messages:
    FamilySubsetsValue: Map from column family name to the columns in this
      family to be included in the AuthorizedView.

  Fields:
    familySubsets: Map from column family name to the columns in this family
      to be included in the AuthorizedView.
    rowPrefixes: Row prefixes to be included in the AuthorizedView. To provide
      access to all rows, include the empty string as a prefix ("").
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FamilySubsetsValue(_messages.Message):
        """Map from column family name to the columns in this family to be
    included in the AuthorizedView.

    Messages:
      AdditionalProperty: An additional property for a FamilySubsetsValue
        object.

    Fields:
      additionalProperties: Additional properties of type FamilySubsetsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FamilySubsetsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleBigtableAdminV2AuthorizedViewFamilySubsets attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleBigtableAdminV2AuthorizedViewFamilySubsets', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    familySubsets = _messages.MessageField('FamilySubsetsValue', 1)
    rowPrefixes = _messages.BytesField(2, repeated=True)