from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MigrationResourcesUserTagsValue(_messages.Message):
    """User specified tags to add to every M2VM generated resource in Azure.
    These tags will be set in addition to the default tags that are set as
    part of the migration process. The tags must not begin with the reserved
    prefix `m4ce` or `m2vm`.

    Messages:
      AdditionalProperty: An additional property for a
        MigrationResourcesUserTagsValue object.

    Fields:
      additionalProperties: Additional properties of type
        MigrationResourcesUserTagsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MigrationResourcesUserTagsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)