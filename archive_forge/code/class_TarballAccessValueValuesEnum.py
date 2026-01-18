from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TarballAccessValueValuesEnum(_messages.Enum):
    """Optional. (Optional) The access type to the diagnostic tarball. If not
    specified, falls back to default access of the bucket

    Values:
      TARBALL_ACCESS_UNSPECIFIED: Tarball Access unspecified. Falls back to
        default access of the bucket
      GOOGLE_CLOUD_SUPPORT: Google Cloud Support group has read access to the
        diagnostic tarball
      GOOGLE_DATAPROC_DIAGNOSE: Google Cloud Dataproc Diagnose service account
        has read access to the diagnostic tarball
    """
    TARBALL_ACCESS_UNSPECIFIED = 0
    GOOGLE_CLOUD_SUPPORT = 1
    GOOGLE_DATAPROC_DIAGNOSE = 2