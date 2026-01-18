from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1alpha2FhirGcsErrorDestination(_messages.Message):
    """Specifies the Cloud Storage destination where errors are recorded.

  Fields:
    uriPrefix: URI for a Cloud Storage directory where the server writes error
      report files, in the format `gs://{bucket-
      id}/{path/to/destination/dir}`. If there is no trailing slash, the
      service appends one when composing the object path. The Cloud Storage
      bucket referenced in `uri_prefix` must exist or an error occurs.
  """
    uriPrefix = _messages.StringField(1)