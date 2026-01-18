from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StudyMetrics(_messages.Message):
    """StudyMetrics contains metrics describing a DICOM study.

  Fields:
    blobStorageSizeBytes: Total blob storage bytes for all instances in the
      study.
    instanceCount: Number of instances in the study.
    seriesCount: Number of series in the study.
    structuredStorageSizeBytes: Total structured storage bytes for all
      instances in the study.
    study: The study resource path. For example, `projects/{project_id}/locati
      ons/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_id}/dic
      omWeb/studies/{study_uid}`.
  """
    blobStorageSizeBytes = _messages.IntegerField(1)
    instanceCount = _messages.IntegerField(2)
    seriesCount = _messages.IntegerField(3)
    structuredStorageSizeBytes = _messages.IntegerField(4)
    study = _messages.StringField(5)