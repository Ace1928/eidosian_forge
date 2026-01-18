from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ComponentVersionValue(_messages.Message):
    """The components that should be installed in this Dataproc cluster. The
    key must be a string from the KubernetesComponent enumeration. The value
    is the version of the software to be installed. At least one entry must be
    specified.

    Messages:
      AdditionalProperty: An additional property for a ComponentVersionValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ComponentVersionValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ComponentVersionValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)