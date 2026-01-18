from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeFleetState(_messages.Message):
    """**ClusterUpgrade**: The state for the fleet-level ClusterUpgrade
  feature.

  Messages:
    IgnoredValue: A list of memberships ignored by the feature. For example,
      manually upgraded clusters can be ignored if they are newer than the
      default versions of its release channel. The membership resource is in
      the format: `projects/{p}/locations/{l}/membership/{m}`.

  Fields:
    downstreamFleets: This fleets whose upstream_fleets contain the current
      fleet. The fleet name should be either fleet project number or id.
    gkeState: Feature state for GKE clusters.
    ignored: A list of memberships ignored by the feature. For example,
      manually upgraded clusters can be ignored if they are newer than the
      default versions of its release channel. The membership resource is in
      the format: `projects/{p}/locations/{l}/membership/{m}`.
  """

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
    downstreamFleets = _messages.StringField(1, repeated=True)
    gkeState = _messages.MessageField('ClusterUpgradeGKEUpgradeFeatureState', 2)
    ignored = _messages.MessageField('IgnoredValue', 3)