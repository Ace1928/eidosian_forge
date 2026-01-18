from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HybridPoolConfig(_messages.Message):
    """Configuration for a Hybrid Worker Pool Next ID: 6

  Enums:
    BuilderImageCachingValueValuesEnum: Immutable. Controls how the worker
      pool caches images. If unspecified during worker pool creation, this
      field is defaulted to CACHING_DISABLED.

  Fields:
    builderImageCaching: Immutable. Controls how the worker pool caches
      images. If unspecified during worker pool creation, this field is
      defaulted to CACHING_DISABLED.
    defaultWorkerConfig: Default settings which will be applied to builds on
      this worker pool if they are not specified in the build request.
    membership: Required. Immutable. The Anthos/GKE Hub membership of the
      cluster which will run the actual build operations. Example:
      projects/{project}/locations/{location}/memberships/{cluster_name}
  """

    class BuilderImageCachingValueValuesEnum(_messages.Enum):
        """Immutable. Controls how the worker pool caches images. If unspecified
    during worker pool creation, this field is defaulted to CACHING_DISABLED.

    Values:
      BUILDER_IMAGE_CACHING_UNSPECIFIED: Default enum type. This should not be
        used.
      CACHING_DISABLED: DinD caching is disabled and no caching resources are
        provisioned.
      VOLUME_CACHING: A PersistentVolumeClaim is provisioned for caching.
    """
        BUILDER_IMAGE_CACHING_UNSPECIFIED = 0
        CACHING_DISABLED = 1
        VOLUME_CACHING = 2
    builderImageCaching = _messages.EnumField('BuilderImageCachingValueValuesEnum', 1)
    defaultWorkerConfig = _messages.MessageField('HybridWorkerConfig', 2)
    membership = _messages.StringField(3)