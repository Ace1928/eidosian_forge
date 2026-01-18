from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsSupportedDatabaseFlagsListRequest(_messages.Message):
    """A AlloydbProjectsLocationsSupportedDatabaseFlagsListRequest object.

  Fields:
    pageSize: Requested page size. Server may return fewer items than
      requested. If unspecified, server will pick an appropriate default.
    pageToken: A token identifying a page of results the server should return.
    parent: Required. The name of the parent resource. The required format is:
      * projects/{project}/locations/{location} Regardless of the parent
      specified here, as long it is contains a valid project and location, the
      service will return a static list of supported flags resources. Note
      that we do not yet support region-specific flags.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)