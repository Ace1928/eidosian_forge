from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1EntryTypeAuthorization(_messages.Message):
    """Authorization for an Entry Type.

  Fields:
    alternateUsePermission: Immutable. The IAM permission grantable on the
      Entry Group to allow access to instantiate Entries of Dataplex owned
      Entry Types, only settable for Dataplex owned Types.
  """
    alternateUsePermission = _messages.StringField(1)