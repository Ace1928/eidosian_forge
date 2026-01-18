from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class IgnoredValue(_messages.Message):
    """A list of memberships ignored by the feature. For example, manually
    upgraded clusters can be ignored if they are newer than the default
    versions of its release channel. The membership resource is in the format:
    `projects/{p}/locations/{l}/membership/{m}`.

    Messages:
      AdditionalProperty: An additional property for a IgnoredValue object.

    Fields:
      additionalProperties: Additional properties of type IgnoredValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a IgnoredValue object.

      Fields:
        key: Name of the additional property.
        value: A ClusterUpgradeIgnoredMembership attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ClusterUpgradeIgnoredMembership', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)