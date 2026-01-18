from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatacatalogProjectsLocationsTagTemplatesDeleteRequest(_messages.Message):
    """A DatacatalogProjectsLocationsTagTemplatesDeleteRequest object.

  Fields:
    force: Required. If true, deletes all tags that use this template.
      Currently, `true` is the only supported value.
    name: Required. The name of the tag template to delete.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)