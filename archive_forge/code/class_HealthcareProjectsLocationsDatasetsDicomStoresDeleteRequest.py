from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsDicomStoresDeleteRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsDicomStoresDeleteRequest object.

  Fields:
    name: Required. The resource name of the DICOM store to delete.
  """
    name = _messages.StringField(1, required=True)