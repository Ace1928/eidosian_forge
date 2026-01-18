from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConfigManagementPolicyControllerVersion(_messages.Message):
    """The build version of Gatekeeper Policy Controller is using.

  Fields:
    version: The gatekeeper image tag that is composed of ACM version, git
      tag, build number.
  """
    version = _messages.StringField(1)