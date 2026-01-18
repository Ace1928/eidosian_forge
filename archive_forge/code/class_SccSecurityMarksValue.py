from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SccSecurityMarksValue(_messages.Message):
    """The actual content of Security Command Center security marks
    associated with the asset. To search against SCC SecurityMarks field: *
    Use a field query: - query by a given key value pair. Example:
    `sccSecurityMarks.foo=bar` - query by a given key's existence. Example:
    `sccSecurityMarks.foo:*`

    Messages:
      AdditionalProperty: An additional property for a SccSecurityMarksValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        SccSecurityMarksValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SccSecurityMarksValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)