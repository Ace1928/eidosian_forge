from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MediaassetProjectsLocationsCatalogsCreateRequest(_messages.Message):
    """A MediaassetProjectsLocationsCatalogsCreateRequest object.

  Fields:
    catalog: A Catalog resource to be passed as the request body.
    catalogId: The ID of the catalog resource to be created.
    parent: Required. The parent resource name, in the following form:
      `projects/{project}/locations/{location}`.
  """
    catalog = _messages.MessageField('Catalog', 1)
    catalogId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)