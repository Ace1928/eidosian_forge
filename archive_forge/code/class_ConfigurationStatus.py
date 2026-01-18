from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigurationStatus(_messages.Message):
    """ConfigurationStatus communicates the observed state of the Configuration
  (from the controller).

  Fields:
    conditions: Conditions communicate information about ongoing/complete
      reconciliation processes that bring the "spec" inline with the observed
      state of the world.
    latestCreatedRevisionName: LatestCreatedRevisionName is the last revision
      that was created from this Configuration. It might not be ready yet, so
      for the latest ready revision, use LatestReadyRevisionName.
    latestReadyRevisionName: LatestReadyRevisionName holds the name of the
      latest Revision stamped out from this Configuration that has had its
      "Ready" condition become "True".
    observedGeneration: ObservedGeneration is the 'Generation' of the
      Configuration that was last processed by the controller. The observed
      generation is updated even if the controller failed to process the spec
      and create the Revision. Clients polling for completed reconciliation
      should poll until observedGeneration = metadata.generation, and the
      Ready condition's status is True or False.
  """
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 1, repeated=True)
    latestCreatedRevisionName = _messages.StringField(2)
    latestReadyRevisionName = _messages.StringField(3)
    observedGeneration = _messages.IntegerField(4, variant=_messages.Variant.INT32)