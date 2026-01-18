from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1RaySpec(_messages.Message):
    """Configuration information for the Ray cluster. For experimental launch,
  Ray cluster creation and Persistent cluster creation are 1:1 mapping: We
  will provision all the nodes within the Persistent cluster as Ray nodes.

  Messages:
    ResourcePoolImagesValue: Optional. Required if image_uri isn't set. A map
      of resource_pool_id to prebuild Ray image if user need to use different
      images for different head/worker pools. This map needs to cover all the
      resource pool ids. Example: { "ray_head_node_pool": "head image"
      "ray_worker_node_pool1": "worker image" "ray_worker_node_pool2":
      "another worker image" }

  Fields:
    headNodeResourcePoolId: Optional. This will be used to indicate which
      resource pool will serve as the Ray head node(the first node within that
      pool). Will use the machine from the first workerpool as the head node
      by default if this field isn't set.
    imageUri: Optional. Default image for user to choose a preferred ML
      framework (for example, TensorFlow or Pytorch) by choosing from [Vertex
      prebuilt images](https://cloud.google.com/vertex-ai/docs/training/pre-
      built-containers). Either this or the resource_pool_images is required.
      Use this field if you need all the resource pools to have the same Ray
      image. Otherwise, use the {@code resource_pool_images} field.
    rayMetricSpec: Optional. Ray metrics configurations.
    resourcePoolImages: Optional. Required if image_uri isn't set. A map of
      resource_pool_id to prebuild Ray image if user need to use different
      images for different head/worker pools. This map needs to cover all the
      resource pool ids. Example: { "ray_head_node_pool": "head image"
      "ray_worker_node_pool1": "worker image" "ray_worker_node_pool2":
      "another worker image" }
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ResourcePoolImagesValue(_messages.Message):
        """Optional. Required if image_uri isn't set. A map of resource_pool_id
    to prebuild Ray image if user need to use different images for different
    head/worker pools. This map needs to cover all the resource pool ids.
    Example: { "ray_head_node_pool": "head image" "ray_worker_node_pool1":
    "worker image" "ray_worker_node_pool2": "another worker image" }

    Messages:
      AdditionalProperty: An additional property for a ResourcePoolImagesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        ResourcePoolImagesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ResourcePoolImagesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    headNodeResourcePoolId = _messages.StringField(1)
    imageUri = _messages.StringField(2)
    rayMetricSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1RayMetricSpec', 3)
    resourcePoolImages = _messages.MessageField('ResourcePoolImagesValue', 4)