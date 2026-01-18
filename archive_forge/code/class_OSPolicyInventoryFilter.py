from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyInventoryFilter(_messages.Message):
    """Filtering criteria to select VMs based on inventory details.

  Fields:
    osShortName: Required. The OS short name
    osVersion: The OS version Prefix matches are supported if asterisk(*) is
      provided as the last character. For example, to match all versions with
      a major version of `7`, specify the following value for this field `7.*`
      An empty string matches all OS versions.
  """
    osShortName = _messages.StringField(1)
    osVersion = _messages.StringField(2)