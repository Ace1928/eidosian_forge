from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirReadRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirReadRequest object.

  Fields:
    name: The name of the resource to retrieve.
  """
    name = _messages.StringField(1, required=True)