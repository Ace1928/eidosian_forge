from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportDicomDataRequest(_messages.Message):
    """Imports data into the specified DICOM store. Returns an error if any of
  the files to import are not DICOM files. This API accepts duplicate DICOM
  instances by ignoring the newly-pushed instance. It does not overwrite.

  Fields:
    gcsSource: Cloud Storage source data location and import configuration.
      The Cloud Healthcare Service Agent requires the
      `roles/storage.objectViewer` Cloud IAM roles on the Cloud Storage
      location.
  """
    gcsSource = _messages.MessageField('GoogleCloudHealthcareV1alpha2DicomGcsSource', 1)