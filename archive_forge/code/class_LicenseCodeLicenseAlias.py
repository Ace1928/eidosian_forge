from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicenseCodeLicenseAlias(_messages.Message):
    """A LicenseCodeLicenseAlias object.

  Fields:
    description: [Output Only] Description of this License Code.
    selfLink: [Output Only] URL of license corresponding to this License Code.
  """
    description = _messages.StringField(1)
    selfLink = _messages.StringField(2)