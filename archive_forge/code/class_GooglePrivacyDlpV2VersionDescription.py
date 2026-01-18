from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2VersionDescription(_messages.Message):
    """Details about each available version for an infotype.

  Fields:
    description: Description of the version.
    version: Name of the version
  """
    description = _messages.StringField(1)
    version = _messages.StringField(2)