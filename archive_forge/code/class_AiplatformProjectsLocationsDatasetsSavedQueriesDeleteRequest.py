from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsSavedQueriesDeleteRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsSavedQueriesDeleteRequest object.

  Fields:
    name: Required. The resource name of the SavedQuery to delete. Format: `pr
      ojects/{project}/locations/{location}/datasets/{dataset}/savedQueries/{s
      aved_query}`
  """
    name = _messages.StringField(1, required=True)