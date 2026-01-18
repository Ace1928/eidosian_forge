from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListExportsResponse(_messages.Message):
    """The response for ListExports

  Fields:
    exports: Details of the export jobs.
  """
    exports = _messages.MessageField('GoogleCloudApigeeV1Export', 1, repeated=True)