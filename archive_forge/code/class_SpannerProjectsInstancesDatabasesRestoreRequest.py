from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesRestoreRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesRestoreRequest object.

  Fields:
    parent: Required. The name of the instance in which to create the restored
      database. This instance must be in the same project and have the same
      instance configuration as the instance containing the source backup.
      Values are of the form `projects//instances/`.
    restoreDatabaseRequest: A RestoreDatabaseRequest resource to be passed as
      the request body.
  """
    parent = _messages.StringField(1, required=True)
    restoreDatabaseRequest = _messages.MessageField('RestoreDatabaseRequest', 2)