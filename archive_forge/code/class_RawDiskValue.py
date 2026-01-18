from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RawDiskValue(_messages.Message):
    """The parameters of the raw disk image.

    Enums:
      ContainerTypeValueValuesEnum: The format used to encode and transmit the
        block device, which should be TAR. This is just a container and
        transmission format and not a runtime format. Provided by the client
        when the disk image is created.

    Fields:
      containerType: The format used to encode and transmit the block device,
        which should be TAR. This is just a container and transmission format
        and not a runtime format. Provided by the client when the disk image
        is created.
      sha1Checksum: [Deprecated] This field is deprecated. An optional SHA1
        checksum of the disk image before unpackaging provided by the client
        when the disk image is created.
      source: The full Google Cloud Storage URL where the raw disk image
        archive is stored. The following are valid formats for the URL: -
        https://storage.googleapis.com/bucket_name/image_archive_name -
        https://storage.googleapis.com/bucket_name/folder_name/
        image_archive_name In order to create an image, you must provide the
        full or partial URL of one of the following: - The rawDisk.source URL
        - The sourceDisk URL - The sourceImage URL - The sourceSnapshot URL
    """

    class ContainerTypeValueValuesEnum(_messages.Enum):
        """The format used to encode and transmit the block device, which
      should be TAR. This is just a container and transmission format and not
      a runtime format. Provided by the client when the disk image is created.

      Values:
        TAR: <no description>
      """
        TAR = 0
    containerType = _messages.EnumField('ContainerTypeValueValuesEnum', 1)
    sha1Checksum = _messages.StringField(2)
    source = _messages.StringField(3)