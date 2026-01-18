from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesDeleteRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesDeleteRequest object.

  Fields:
    name: Required. Resource name of the taxonomy to be deleted. All
      categories in this taxonomy will also be deleted.
  """
    name = _messages.StringField(1, required=True)