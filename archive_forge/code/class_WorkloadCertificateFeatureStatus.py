from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadCertificateFeatureStatus(_messages.Message):
    """Status of Workload Certificate feature at trust domain level.

  Enums:
    StateValueValuesEnum: Describes whether the Workload Certificate feature
      meets its spec.

  Messages:
    ManagedCaPoolsValue: A map from a region to the status of managed CA pools
      in that region.

  Fields:
    managedCaPools: A map from a region to the status of managed CA pools in
      that region.
    state: Describes whether the Workload Certificate feature meets its spec.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Describes whether the Workload Certificate feature meets its spec.

    Values:
      FEATURE_STATE_UNSPECIFIED: The feature status does not fully meet its
        spec at the moment but is trying to meet its spec.
      FEATURE_STATE_IN_PROGRESS: The feature status does not fully meet its
        spec at the moment but is trying to meet its spec.
      FEATURE_STATE_READY: The feature status currently meets its spec.
      FEATURE_STATE_INTERNAL_ERROR: The feature status does not fully meet its
        spec at the moment due to an internal error but the backend is
        retrying. Contact support if this persists.
    """
        FEATURE_STATE_UNSPECIFIED = 0
        FEATURE_STATE_IN_PROGRESS = 1
        FEATURE_STATE_READY = 2
        FEATURE_STATE_INTERNAL_ERROR = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ManagedCaPoolsValue(_messages.Message):
        """A map from a region to the status of managed CA pools in that region.

    Messages:
      AdditionalProperty: An additional property for a ManagedCaPoolsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ManagedCaPoolsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ManagedCaPoolsValue object.

      Fields:
        key: Name of the additional property.
        value: A CaPoolsStatus attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('CaPoolsStatus', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    managedCaPools = _messages.MessageField('ManagedCaPoolsValue', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)