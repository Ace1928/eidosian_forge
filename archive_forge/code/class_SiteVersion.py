from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SiteVersion(_messages.Message):
    """SiteVersion Hydration is targeting.

  Fields:
    nfType: Output only. NF vendor type.
    nfVendor: Output only. NF vendor.
    nfVersion: Output only. NF version.
  """
    nfType = _messages.StringField(1)
    nfVendor = _messages.StringField(2)
    nfVersion = _messages.StringField(3)