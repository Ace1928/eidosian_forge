from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapProjectsBrandsCreateRequest(_messages.Message):
    """A IapProjectsBrandsCreateRequest object.

  Fields:
    brand: A Brand resource to be passed as the request body.
    parent: Required. GCP Project number/id under which the brand is to be
      created. In the following format: projects/{project_number/id}.
  """
    brand = _messages.MessageField('Brand', 1)
    parent = _messages.StringField(2, required=True)