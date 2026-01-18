from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1DataAttributeBindingPath(_messages.Message):
    """Represents a subresource of the given resource, and associated bindings
  with it. Currently supported subresources are column and partition schema
  fields within a table.

  Fields:
    attributes: Optional. List of attributes to be associated with the path of
      the resource, provided in the form: projects/{project}/locations/{locati
      on}/dataTaxonomies/{dataTaxonomy}/attributes/{data_attribute_id}
    name: Required. The name identifier of the path. Nested columns should be
      of the form: 'address.city'.
  """
    attributes = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2)