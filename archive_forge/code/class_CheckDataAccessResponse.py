from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckDataAccessResponse(_messages.Message):
    """Checks if a particular data_id of a User data mapping in the given
  consent store is consented for a given use.

  Messages:
    ConsentDetailsValue: The resource names of all evaluated Consents mapped
      to their evaluation.

  Fields:
    consentDetails: The resource names of all evaluated Consents mapped to
      their evaluation.
    consented: Whether the requested resource is consented for the given use.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ConsentDetailsValue(_messages.Message):
        """The resource names of all evaluated Consents mapped to their
    evaluation.

    Messages:
      AdditionalProperty: An additional property for a ConsentDetailsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ConsentDetailsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ConsentDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A ConsentEvaluation attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('ConsentEvaluation', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    consentDetails = _messages.MessageField('ConsentDetailsValue', 1)
    consented = _messages.BooleanField(2)