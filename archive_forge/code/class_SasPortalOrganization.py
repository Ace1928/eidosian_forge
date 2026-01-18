from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalOrganization(_messages.Message):
    """Organization details.

  Fields:
    displayName: Name of organization
    id: Id of organization
  """
    displayName = _messages.StringField(1)
    id = _messages.IntegerField(2)