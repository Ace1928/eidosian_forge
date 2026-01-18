from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitServicePerimetersRequest(_messages.Message):
    """A request to commit dry-run specs in all Service Perimeters belonging to
  an Access Policy.

  Fields:
    etag: Optional. The etag for the version of the Access Policy that this
      commit operation is to be performed on. If, at the time of commit, the
      etag for the Access Policy stored in Access Context Manager is different
      from the specified etag, then the commit operation will not be performed
      and the call will fail. This field is not required. If etag is not
      provided, the operation will be performed as if a valid etag is
      provided.
  """
    etag = _messages.StringField(1)