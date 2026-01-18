from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsDataTaxonomiesAttributesGetRequest(_messages.Message):
    """A DataplexProjectsLocationsDataTaxonomiesAttributesGetRequest object.

  Fields:
    name: Required. The resource name of the dataAttribute: projects/{project_
      number}/locations/{location_id}/dataTaxonomies/{dataTaxonomy}/attributes
      /{data_attribute_id}
  """
    name = _messages.StringField(1, required=True)