from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstancesDatabasesPatchRequest(_messages.Message):
    """A SpannerProjectsInstancesDatabasesPatchRequest object.

  Fields:
    database: A Database resource to be passed as the request body.
    name: Required. The name of the database. Values are of the form
      `projects//instances//databases/`, where `` is as specified in the
      `CREATE DATABASE` statement. This name can be passed to other API
      methods to identify the database.
    updateMask: Required. The list of fields to update. Currently, only
      `enable_drop_protection` field can be updated.
  """
    database = _messages.MessageField('Database', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)