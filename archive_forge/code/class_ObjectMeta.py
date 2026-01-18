from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectMeta(_messages.Message):
    """google.cloud.run.meta.v1.ObjectMeta is metadata that all persisted
  resources must have, which includes all objects users must create.

  Messages:
    AnnotationsValue: Unstructured key value map stored with a resource that
      may be set by external tools to store and retrieve arbitrary metadata.
      They are not queryable and should be preserved when modifying objects.
      In Cloud Run, annotations with 'run.googleapis.com/' and
      'autoscaling.knative.dev' are restricted, and the accepted annotations
      will be different depending on the resource type. *
      `autoscaling.knative.dev/maxScale`: Revision. *
      `autoscaling.knative.dev/minScale`: Revision. *
      `run.googleapis.com/binary-authorization-breakglass`: Service, Job, *
      `run.googleapis.com/binary-authorization`: Service, Job, Execution. *
      `run.googleapis.com/client-name`: All resources. *
      `run.googleapis.com/cloudsql-instances`: Revision, Execution. *
      `run.googleapis.com/container-dependencies`: Revision . *
      `run.googleapis.com/cpu-throttling`: Revision. *
      `run.googleapis.com/custom-audiences`: Service. *
      `run.googleapis.com/default-url-disabled`: Service. *
      `run.googleapis.com/description`: Service. *
      `run.googleapis.com/encryption-key-shutdown-hours`: Revision *
      `run.googleapis.com/encryption-key`: Revision, Execution. *
      `run.googleapis.com/execution-environment`: Revision, Execution. *
      `run.googleapis.com/gc-traffic-tags`: Service. *
      `run.googleapis.com/ingress`: Service. * `run.googleapis.com/launch-
      stage`: Service, Job. * `run.googleapis.com/minScale`: Service (ALPHA) *
      `run.googleapis.com/network-interfaces`: Revision, Execution. *
      `run.googleapis.com/post-key-revocation-action-type`: Revision. *
      `run.googleapis.com/secrets`: Revision, Execution. *
      `run.googleapis.com/secure-session-agent`: Revision. *
      `run.googleapis.com/sessionAffinity`: Revision. *
      `run.googleapis.com/startup-cpu-boost`: Revision. *
      `run.googleapis.com/vpc-access-connector`: Revision, Execution. *
      `run.googleapis.com/vpc-access-egress`: Revision, Execution.
    LabelsValue: Map of string keys and values that can be used to organize
      and categorize (scope and select) objects. May match selectors of
      replication controllers and routes.

  Fields:
    annotations: Unstructured key value map stored with a resource that may be
      set by external tools to store and retrieve arbitrary metadata. They are
      not queryable and should be preserved when modifying objects. In Cloud
      Run, annotations with 'run.googleapis.com/' and
      'autoscaling.knative.dev' are restricted, and the accepted annotations
      will be different depending on the resource type. *
      `autoscaling.knative.dev/maxScale`: Revision. *
      `autoscaling.knative.dev/minScale`: Revision. *
      `run.googleapis.com/binary-authorization-breakglass`: Service, Job, *
      `run.googleapis.com/binary-authorization`: Service, Job, Execution. *
      `run.googleapis.com/client-name`: All resources. *
      `run.googleapis.com/cloudsql-instances`: Revision, Execution. *
      `run.googleapis.com/container-dependencies`: Revision . *
      `run.googleapis.com/cpu-throttling`: Revision. *
      `run.googleapis.com/custom-audiences`: Service. *
      `run.googleapis.com/default-url-disabled`: Service. *
      `run.googleapis.com/description`: Service. *
      `run.googleapis.com/encryption-key-shutdown-hours`: Revision *
      `run.googleapis.com/encryption-key`: Revision, Execution. *
      `run.googleapis.com/execution-environment`: Revision, Execution. *
      `run.googleapis.com/gc-traffic-tags`: Service. *
      `run.googleapis.com/ingress`: Service. * `run.googleapis.com/launch-
      stage`: Service, Job. * `run.googleapis.com/minScale`: Service (ALPHA) *
      `run.googleapis.com/network-interfaces`: Revision, Execution. *
      `run.googleapis.com/post-key-revocation-action-type`: Revision. *
      `run.googleapis.com/secrets`: Revision, Execution. *
      `run.googleapis.com/secure-session-agent`: Revision. *
      `run.googleapis.com/sessionAffinity`: Revision. *
      `run.googleapis.com/startup-cpu-boost`: Revision. *
      `run.googleapis.com/vpc-access-connector`: Revision, Execution. *
      `run.googleapis.com/vpc-access-egress`: Revision, Execution.
    clusterName: Not supported by Cloud Run
    creationTimestamp: UTC timestamp representing the server time when this
      object was created.
    deletionGracePeriodSeconds: Not supported by Cloud Run
    deletionTimestamp: The read-only soft deletion timestamp for this
      resource. In Cloud Run, users are not able to set this field. Instead,
      they must call the corresponding Delete API.
    finalizers: Not supported by Cloud Run
    generateName: Not supported by Cloud Run
    generation: A system-provided sequence number representing a specific
      generation of the desired state.
    labels: Map of string keys and values that can be used to organize and
      categorize (scope and select) objects. May match selectors of
      replication controllers and routes.
    name: Required. The name of the resource. Name is required when creating
      top-level resources (Service, Job), must be unique within a Cloud Run
      project/region, and cannot be changed once created.
    namespace: Required. Defines the space within each name must be unique
      within a Cloud Run region. In Cloud Run, it must be project ID or
      number.
    ownerReferences: Not supported by Cloud Run
    resourceVersion: Opaque, system-generated value that represents the
      internal version of this object that can be used by clients to determine
      when objects have changed. May be used for optimistic concurrency,
      change detection, and the watch operation on a resource or set of
      resources. Clients must treat these values as opaque and passed
      unmodified back to the server or omit the value to disable conflict-
      detection.
    selfLink: URL representing this object.
    uid: Unique, system-generated identifier for this resource.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Unstructured key value map stored with a resource that may be set by
    external tools to store and retrieve arbitrary metadata. They are not
    queryable and should be preserved when modifying objects. In Cloud Run,
    annotations with 'run.googleapis.com/' and 'autoscaling.knative.dev' are
    restricted, and the accepted annotations will be different depending on
    the resource type. * `autoscaling.knative.dev/maxScale`: Revision. *
    `autoscaling.knative.dev/minScale`: Revision. *
    `run.googleapis.com/binary-authorization-breakglass`: Service, Job, *
    `run.googleapis.com/binary-authorization`: Service, Job, Execution. *
    `run.googleapis.com/client-name`: All resources. *
    `run.googleapis.com/cloudsql-instances`: Revision, Execution. *
    `run.googleapis.com/container-dependencies`: Revision . *
    `run.googleapis.com/cpu-throttling`: Revision. *
    `run.googleapis.com/custom-audiences`: Service. *
    `run.googleapis.com/default-url-disabled`: Service. *
    `run.googleapis.com/description`: Service. *
    `run.googleapis.com/encryption-key-shutdown-hours`: Revision *
    `run.googleapis.com/encryption-key`: Revision, Execution. *
    `run.googleapis.com/execution-environment`: Revision, Execution. *
    `run.googleapis.com/gc-traffic-tags`: Service. *
    `run.googleapis.com/ingress`: Service. * `run.googleapis.com/launch-
    stage`: Service, Job. * `run.googleapis.com/minScale`: Service (ALPHA) *
    `run.googleapis.com/network-interfaces`: Revision, Execution. *
    `run.googleapis.com/post-key-revocation-action-type`: Revision. *
    `run.googleapis.com/secrets`: Revision, Execution. *
    `run.googleapis.com/secure-session-agent`: Revision. *
    `run.googleapis.com/sessionAffinity`: Revision. *
    `run.googleapis.com/startup-cpu-boost`: Revision. *
    `run.googleapis.com/vpc-access-connector`: Revision, Execution. *
    `run.googleapis.com/vpc-access-egress`: Revision, Execution.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Map of string keys and values that can be used to organize and
    categorize (scope and select) objects. May match selectors of replication
    controllers and routes.

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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    clusterName = _messages.StringField(2)
    creationTimestamp = _messages.StringField(3)
    deletionGracePeriodSeconds = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    deletionTimestamp = _messages.StringField(5)
    finalizers = _messages.StringField(6, repeated=True)
    generateName = _messages.StringField(7)
    generation = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    labels = _messages.MessageField('LabelsValue', 9)
    name = _messages.StringField(10)
    namespace = _messages.StringField(11)
    ownerReferences = _messages.MessageField('OwnerReference', 12, repeated=True)
    resourceVersion = _messages.StringField(13)
    selfLink = _messages.StringField(14)
    uid = _messages.StringField(15)