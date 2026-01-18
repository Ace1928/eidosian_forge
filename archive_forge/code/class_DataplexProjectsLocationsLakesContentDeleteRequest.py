from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsLakesContentDeleteRequest(_messages.Message):
    """A DataplexProjectsLocationsLakesContentDeleteRequest object.

  Fields:
    name: Required. The resource name of the content: projects/{project_id}/lo
      cations/{location_id}/lakes/{lake_id}/content/{content_id}
  """
    name = _messages.StringField(1, required=True)