from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class PypiDependenciesValue(_messages.Message):
    """Pypi dependencies specified in the environment configuration, at the
    time when the build was triggered.

    Messages:
      AdditionalProperty: An additional property for a PypiDependenciesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PypiDependenciesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a PypiDependenciesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)