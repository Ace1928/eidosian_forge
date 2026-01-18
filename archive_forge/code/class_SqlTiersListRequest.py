from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SqlTiersListRequest(_messages.Message):
    """A SqlTiersListRequest object.

  Fields:
    project: Project ID of the project for which to list tiers.
  """
    project = _messages.StringField(1, required=True)