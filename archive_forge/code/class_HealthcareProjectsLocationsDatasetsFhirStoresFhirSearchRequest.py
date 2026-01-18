from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsFhirStoresFhirSearchRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsFhirStoresFhirSearchRequest object.

  Fields:
    parent: Name of the FHIR store to retrieve resources from.
    searchResourcesRequest: A SearchResourcesRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    searchResourcesRequest = _messages.MessageField('SearchResourcesRequest', 2)