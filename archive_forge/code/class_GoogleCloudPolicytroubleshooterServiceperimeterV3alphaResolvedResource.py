from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResolvedResource(_messages.Message):
    """The details of a resolved resource NextTAG: 7

  Enums:
    ResolvedStateValueValuesEnum: The resolved resource's state

  Fields:
    bridgeServicePerimeters: Full resource names of the bridge service
      perimeters that restrict the resource Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`
    dryrunBridgeServicePerimeters: Full resource names of the dryrun bridge
      service perimeters that restrict the resource Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`
    dryrunRegularServicePerimeters: Full resource name of the dry run regular
      service perimeters that restricts the resource Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`
    regularServicePerimeters: Full resource name of the regular service
      perimeters that restricts the resource Format:
      `accessPolicies/{access_policy}/servicePerimeters/{service_perimeter}`
    resolvedState: The resolved resource's state
    resource: Details of the resource
  """

    class ResolvedStateValueValuesEnum(_messages.Enum):
        """The resolved resource's state

    Values:
      RESOLVED_STATE_UNSPECIFIED: Not used
      INFO_DENIED: The caller doesn't have permission to resolve this resource
      COMPLETED: The resource has been fully resolved
      NOT_APPLICABLE: The resource cannot be restricted by service perimeters
      ERROR: The resource cannot be resolved due to an error.
    """
        RESOLVED_STATE_UNSPECIFIED = 0
        INFO_DENIED = 1
        COMPLETED = 2
        NOT_APPLICABLE = 3
        ERROR = 4
    bridgeServicePerimeters = _messages.StringField(1, repeated=True)
    dryrunBridgeServicePerimeters = _messages.StringField(2, repeated=True)
    dryrunRegularServicePerimeters = _messages.StringField(3, repeated=True)
    regularServicePerimeters = _messages.StringField(4, repeated=True)
    resolvedState = _messages.EnumField('ResolvedStateValueValuesEnum', 5)
    resource = _messages.MessageField('GoogleCloudPolicytroubleshooterServiceperimeterV3alphaResource', 6)