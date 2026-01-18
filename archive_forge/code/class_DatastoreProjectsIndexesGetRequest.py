from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatastoreProjectsIndexesGetRequest(_messages.Message):
    """A DatastoreProjectsIndexesGetRequest object.

  Fields:
    indexId: The resource ID of the index to get.
    projectId: Project ID against which to make the request.
  """
    indexId = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)