from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class TablesValue(_messages.Message):
    """Keys: the full `name` of each table that existed in the instance when
    CreateCluster was first called, i.e. `projects//instances//tables/`. Any
    table added to the instance by a later API call will be created in the new
    cluster by that API call, not this one. Values: information on how much of
    a table's data has been copied to the newly-created cluster so far.

    Messages:
      AdditionalProperty: An additional property for a TablesValue object.

    Fields:
      additionalProperties: Additional properties of type TablesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a TablesValue object.

      Fields:
        key: Name of the additional property.
        value: A TableProgress attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TableProgress', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)