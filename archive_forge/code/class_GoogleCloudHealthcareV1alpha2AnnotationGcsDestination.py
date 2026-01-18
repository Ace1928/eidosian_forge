from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1alpha2AnnotationGcsDestination(_messages.Message):
    """The Cloud Storage location for export.

  Fields:
    uriPrefix: The Cloud Storage destination to export to. URI for a Cloud
      Storage directory where the server writes result files, in the format
      `gs://{bucket-id}/{path/to/destination/dir}`. If there is no trailing
      slash, the service appends one when composing the object path. The user
      is responsible for creating the Cloud Storage bucket referenced in
      `uri_prefix`.
  """
    uriPrefix = _messages.StringField(1)