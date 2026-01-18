from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesCategoriesDeleteRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesCategoriesDeleteRequest object.

  Fields:
    name: Required. Resource name of the category to be deleted. All its
      descendant categories will also be deleted.
  """
    name = _messages.StringField(1, required=True)