from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesGetStudyMetricsRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDicomWebStudiesGetStudyM
  etricsRequest object.

  Fields:
    study: The study resource path. For example, `projects/{project_id}/locati
      ons/{location_id}/datasets/{dataset_id}/dicomStores/{dicom_store_id}/dic
      omWeb/studies/{study_uid}`.
  """
    study = _messages.StringField(1, required=True)