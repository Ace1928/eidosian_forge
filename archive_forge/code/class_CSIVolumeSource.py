from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CSIVolumeSource(_messages.Message):
    """Storage volume source using the Container Storage Interface.

  Messages:
    VolumeAttributesValue: stores driver specific attributes. For Google Cloud
      Storage volumes, the following attributes are supported: * bucketName:
      the name of the Cloud Storage bucket to mount. The Cloud Run Service
      identity must have access to this bucket.

  Fields:
    driver: name of the CSI driver for the requested storage system. Cloud Run
      supports the following drivers: * gcsfuse.run.googleapis.com : Mount a
      Cloud Storage Bucket as a volume.
    readOnly: If true, mount the volume as read only. Defaults to false.
    volumeAttributes: stores driver specific attributes. For Google Cloud
      Storage volumes, the following attributes are supported: * bucketName:
      the name of the Cloud Storage bucket to mount. The Cloud Run Service
      identity must have access to this bucket.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class VolumeAttributesValue(_messages.Message):
        """stores driver specific attributes. For Google Cloud Storage volumes,
    the following attributes are supported: * bucketName: the name of the
    Cloud Storage bucket to mount. The Cloud Run Service identity must have
    access to this bucket.

    Messages:
      AdditionalProperty: An additional property for a VolumeAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        VolumeAttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a VolumeAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    driver = _messages.StringField(1)
    readOnly = _messages.BooleanField(2)
    volumeAttributes = _messages.MessageField('VolumeAttributesValue', 3)