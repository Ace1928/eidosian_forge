from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BackendMetastoresValue(_messages.Message):
    """A map from BackendMetastore rank to BackendMetastores from which the
    federation service serves metadata at query time. The map key represents
    the order in which BackendMetastores should be evaluated to resolve
    database names at query time and should be greater than or equal to zero.
    A BackendMetastore with a lower number will be evaluated before a
    BackendMetastore with a higher number.

    Messages:
      AdditionalProperty: An additional property for a BackendMetastoresValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        BackendMetastoresValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BackendMetastoresValue object.

      Fields:
        key: Name of the additional property.
        value: A BackendMetastore attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('BackendMetastore', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)