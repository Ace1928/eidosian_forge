from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesDeleteRequest object.

  Fields:
    name: Required. The resource name of the lake:
      projects/{project_number}/locations/{location_id}/lakes/{lake_id}.
  """
    name = _messages.StringField(1, required=True)