from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApplianceVersion(_messages.Message):
    """Describes an appliance version.

  Fields:
    critical: Determine whether it's critical to upgrade the appliance to this
      version.
    releaseNotesUri: Link to a page that contains the version release notes.
    uri: A link for downloading the version.
    version: The appliance version.
  """
    critical = _messages.BooleanField(1)
    releaseNotesUri = _messages.StringField(2)
    uri = _messages.StringField(3)
    version = _messages.StringField(4)