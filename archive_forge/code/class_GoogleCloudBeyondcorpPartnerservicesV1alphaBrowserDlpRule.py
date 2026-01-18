from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpPartnerservicesV1alphaBrowserDlpRule(_messages.Message):
    """Browser DLP Rule for a PartnerTenant

  Fields:
    group: Required. The group to which this Rule should be applied to.
    name: Output only. Unique resource name. The name is ignored when creating
      BrowserDlpRule.
    ruleSetting: Required. The policy settings to apply.
  """
    group = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaGroup', 1)
    name = _messages.StringField(2)
    ruleSetting = _messages.MessageField('GoogleCloudBeyondcorpPartnerservicesV1alphaRuleSetting', 3)