from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsGetRequest(_messages.Message):
    """A IapProjectsBrandsGetRequest object.

  Fields:
    name: Required. Name of the brand to be fetched. In the following format:
      projects/{project_number/id}/brands/{brand}.
  """
    name = _messages.StringField(1, required=True)