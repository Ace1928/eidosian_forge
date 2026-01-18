from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkservicesProjectsLocationsHttpFiltersCreateRequest(_messages.Message):
    """A NetworkservicesProjectsLocationsHttpFiltersCreateRequest object.

  Fields:
    httpFilter: A HttpFilter resource to be passed as the request body.
    httpFilterId: Required. Short name of the HttpFilter resource to be
      created. E.g. "CustomFilter".
    parent: Required. The parent resource of the HttpFilter. Must be in the
      format `projects/*/locations/global`.
  """
    httpFilter = _messages.MessageField('HttpFilter', 1)
    httpFilterId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)