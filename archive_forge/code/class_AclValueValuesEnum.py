from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AclValueValuesEnum(_messages.Enum):
    """Specifies how each object's ACLs should be preserved for transfers
    between Google Cloud Storage buckets. If unspecified, the default behavior
    is the same as ACL_DESTINATION_BUCKET_DEFAULT.

    Values:
      ACL_UNSPECIFIED: ACL behavior is unspecified.
      ACL_DESTINATION_BUCKET_DEFAULT: Use the destination bucket's default
        object ACLS, if applicable.
      ACL_PRESERVE: Preserve the object's original ACLs. This requires the
        service account to have `storage.objects.getIamPolicy` permission for
        the source object. [Uniform bucket-level
        access](https://cloud.google.com/storage/docs/uniform-bucket-level-
        access) must not be enabled on either the source or destination
        buckets.
    """
    ACL_UNSPECIFIED = 0
    ACL_DESTINATION_BUCKET_DEFAULT = 1
    ACL_PRESERVE = 2