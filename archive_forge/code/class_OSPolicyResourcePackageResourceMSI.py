from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceMSI(_messages.Message):
    """An MSI package. MSI packages only support INSTALLED state.

  Fields:
    properties: Additional properties to use during installation. This should
      be in the format of Property=Setting. Appended to the defaults of
      `ACTION=INSTALL REBOOT=ReallySuppress`.
    source: Required. The MSI package.
  """
    properties = _messages.StringField(1, repeated=True)
    source = _messages.MessageField('OSPolicyResourceFile', 2)