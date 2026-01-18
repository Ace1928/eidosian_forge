from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevisionStatus(_messages.Message):
    """RevisionStatus communicates the observed state of the Revision (from the
  controller).

  Fields:
    conditions: Conditions communicate information about ongoing/complete
      reconciliation processes that bring the "spec" inline with the observed
      state of the world. As a Revision is being prepared, it will
      incrementally update conditions. Revision-specific conditions include: *
      `ResourcesAvailable`: `True` when underlying resources have been
      provisioned. * `ContainerHealthy`: `True` when the Revision readiness
      check completes. * `Active`: `True` when the Revision may receive
      traffic.
    desiredReplicas: Output only. The configured number of instances running
      this revision. For Cloud Run, this only includes instances provisioned
      using the minScale annotation. It does not include instances created by
      autoscaling.
    imageDigest: ImageDigest holds the resolved digest for the image specified
      within .Spec.Container.Image. The digest is resolved during the creation
      of Revision. This field holds the digest value regardless of whether a
      tag or digest was originally specified in the Container object.
    logUrl: Optional. Specifies the generated logging url for this particular
      revision based on the revision url template specified in the
      controller's config.
    observedGeneration: ObservedGeneration is the 'Generation' of the Revision
      that was last processed by the controller. Clients polling for completed
      reconciliation should poll until observedGeneration =
      metadata.generation, and the Ready condition's status is True or False.
    serviceName: Not currently used by Cloud Run.
  """
    conditions = _messages.MessageField('GoogleCloudRunV1Condition', 1, repeated=True)
    desiredReplicas = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    imageDigest = _messages.StringField(3)
    logUrl = _messages.StringField(4)
    observedGeneration = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    serviceName = _messages.StringField(6)