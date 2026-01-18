from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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