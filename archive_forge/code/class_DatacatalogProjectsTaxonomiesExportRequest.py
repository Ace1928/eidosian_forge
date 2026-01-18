from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsTaxonomiesExportRequest(_messages.Message):
    """A DatacatalogProjectsTaxonomiesExportRequest object.

  Fields:
    parent: Required. Resource name of the project that taxonomies to be
      exported will share.
    taxonomyNames: Required. Resource names of the taxonomies to be exported.
  """
    parent = _messages.StringField(1, required=True)
    taxonomyNames = _messages.StringField(2, repeated=True)