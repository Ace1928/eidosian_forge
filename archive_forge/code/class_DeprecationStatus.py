from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeprecationStatus(_messages.Message):
    """Deprecation status for a public resource.

  Enums:
    StateValueValuesEnum: The deprecation state of this resource. This can be
      ACTIVE, DEPRECATED, OBSOLETE, or DELETED. Operations which communicate
      the end of life date for an image, can use ACTIVE. Operations which
      create a new resource using a DEPRECATED resource will return
      successfully, but with a warning indicating the deprecated resource and
      recommending its replacement. Operations which use OBSOLETE or DELETED
      resources will be rejected and result in an error.

  Fields:
    deleted: An optional RFC3339 timestamp on or after which the state of this
      resource is intended to change to DELETED. This is only informational
      and the status will not change unless the client explicitly changes it.
    deprecated: An optional RFC3339 timestamp on or after which the state of
      this resource is intended to change to DEPRECATED. This is only
      informational and the status will not change unless the client
      explicitly changes it.
    obsolete: An optional RFC3339 timestamp on or after which the state of
      this resource is intended to change to OBSOLETE. This is only
      informational and the status will not change unless the client
      explicitly changes it.
    replacement: The URL of the suggested replacement for a deprecated
      resource. The suggested replacement resource must be the same kind of
      resource as the deprecated resource.
    state: The deprecation state of this resource. This can be ACTIVE,
      DEPRECATED, OBSOLETE, or DELETED. Operations which communicate the end
      of life date for an image, can use ACTIVE. Operations which create a new
      resource using a DEPRECATED resource will return successfully, but with
      a warning indicating the deprecated resource and recommending its
      replacement. Operations which use OBSOLETE or DELETED resources will be
      rejected and result in an error.
    stateOverride: The rollout policy for this deprecation. This policy is
      only enforced by image family views. The rollout policy restricts the
      zones where the associated resource is considered in a deprecated state.
      When the rollout policy does not include the user specified zone, or if
      the zone is rolled out, the associated resource is considered in a
      deprecated state. The rollout policy for this deprecation is read-only,
      except for allowlisted users. This field might not be configured. To
      view the latest non-deprecated image in a specific zone, use the
      imageFamilyViews.get method.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The deprecation state of this resource. This can be ACTIVE,
    DEPRECATED, OBSOLETE, or DELETED. Operations which communicate the end of
    life date for an image, can use ACTIVE. Operations which create a new
    resource using a DEPRECATED resource will return successfully, but with a
    warning indicating the deprecated resource and recommending its
    replacement. Operations which use OBSOLETE or DELETED resources will be
    rejected and result in an error.

    Values:
      ACTIVE: <no description>
      DELETED: <no description>
      DEPRECATED: <no description>
      OBSOLETE: <no description>
    """
        ACTIVE = 0
        DELETED = 1
        DEPRECATED = 2
        OBSOLETE = 3
    deleted = _messages.StringField(1)
    deprecated = _messages.StringField(2)
    obsolete = _messages.StringField(3)
    replacement = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    stateOverride = _messages.MessageField('RolloutPolicy', 6)