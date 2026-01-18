from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsServiceBindingsCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsServiceBindingsCreateRequest object.

  Fields:
    parent: Required. The parent resource of the ServiceBinding. Must be in
      the format `projects/*/locations/global`.
    serviceBinding: A ServiceBinding resource to be passed as the request
      body.
    serviceBindingId: Required. Short name of the ServiceBinding resource to
      be created.
  """
    parent = _messages.StringField(1, required=True)
    serviceBinding = _messages.MessageField('ServiceBinding', 2)
    serviceBindingId = _messages.StringField(3)