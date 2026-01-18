from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FleetPackage(_messages.Message):
    """message describing FleetPackage object

  Enums:
    DeletionPropagationPolicyValueValuesEnum: Optional. Deletion propagation
      policy for the fleet package.
    StateValueValuesEnum: Optional. The desired state of the fleet package.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    deletionPropagationPolicy: Optional. Deletion propagation policy for the
      fleet package.
    info: Output only. info contains the rollout status of the resource bundle
      across given target clusters.
    labels: Optional. Labels as key value pairs
    name: Identifier. Name of the FleetPackage. Format is
      projects/{project}/locations/
      {location}/fleetPackages/{fleetPackage}/a-z{0,62}
    resourceBundleSelector: Required. resource_bundle_selector determines what
      resource bundle to deploy.
    rolloutStrategy: Optional. The strategy for rolling out resource bundles
      to clusters.
    state: Optional. The desired state of the fleet package.
    target: Optional. The target into which the resource bundle should be
      installed.
    updateTime: Output only. [Output only] Update time stamp
    variantSelector: Required. variant_selector specifies how to select a
      resource bundle variant for a target cluster.
  """

    class DeletionPropagationPolicyValueValuesEnum(_messages.Enum):
        """Optional. Deletion propagation policy for the fleet package.

    Values:
      DELETION_PROPAGATION_POLICY_UNSPECIFIED: Unspecified deletion
        propagation policy. Defaults to FOREGROUND.
      FOREGROUND: Foreground deletion propagation policy. Any resources synced
        to the cluster will be deleted.
      ORPHAN: Orphan deletion propagation policy. Any resources synced to the
        cluster will be abandoned.
    """
        DELETION_PROPAGATION_POLICY_UNSPECIFIED = 0
        FOREGROUND = 1
        ORPHAN = 2

    class StateValueValuesEnum(_messages.Enum):
        """Optional. The desired state of the fleet package.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      ACTIVE: Fleet package is intended to be active.
      SUSPENDED: Fleet package is intended to be suspended.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        SUSPENDED = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    deletionPropagationPolicy = _messages.EnumField('DeletionPropagationPolicyValueValuesEnum', 2)
    info = _messages.MessageField('FleetPackageInfo', 3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    resourceBundleSelector = _messages.MessageField('ResourceBundleSelector', 6)
    rolloutStrategy = _messages.MessageField('RolloutStrategy', 7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    target = _messages.MessageField('Target', 9)
    updateTime = _messages.StringField(10)
    variantSelector = _messages.MessageField('VariantSelector', 11)