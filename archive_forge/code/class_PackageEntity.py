from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PackageEntity(_messages.Message):
    """Package's parent is a schema.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    customFeatures: Custom engine specific features.
    packageBody: The SQL code which creates the package body. If the package
      specification has cursors or subprograms, then the package body is
      mandatory.
    packageSqlCode: The SQL code which creates the package.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 1)
    packageBody = _messages.StringField(2)
    packageSqlCode = _messages.StringField(3)