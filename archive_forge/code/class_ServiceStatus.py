from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceStatus(_messages.Message):
    """The current state of the Service. Output only.

  Fields:
    address: Similar to url, information on where the service is available on
      HTTP.
    conditions: Conditions communicate information about ongoing/complete
      reconciliation processes that bring the `spec` inline with the observed
      state of the world. Service-specific conditions include: *
      `ConfigurationsReady`: `True` when the underlying Configuration is
      ready. * `RoutesReady`: `True` when the underlying Route is ready. *
      `Ready`: `True` when all underlying resources are ready.
    latestCreatedRevisionName: Name of the last revision that was created from
      this Service's Configuration. It might not be ready yet, for that use
      LatestReadyRevisionName.
    latestReadyRevisionName: Name of the latest Revision from this Service's
      Configuration that has had its `Ready` condition become `True`.
    observedGeneration: Returns the generation last seen by the system.
      Clients polling for completed reconciliation should poll until
      observedGeneration = metadata.generation and the Ready condition's
      status is True or False.
    traffic: Holds the configured traffic distribution. These entries will
      always contain RevisionName references. When ConfigurationName appears
      in the spec, this will hold the LatestReadyRevisionName that we last
      observed.
    url: URL that will distribute traffic over the provided traffic targets.
      It generally has the form `https://{route-hash}-{project-hash}-{cluster-
      level-suffix}.a.run.app`
  """
    address = _messages.MessageField('Addressable', 1)
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 2, repeated=True)
    latestCreatedRevisionName = _messages.StringField(3)
    latestReadyRevisionName = _messages.StringField(4)
    observedGeneration = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    traffic = _messages.MessageField('TrafficTarget', 6, repeated=True)
    url = _messages.StringField(7)