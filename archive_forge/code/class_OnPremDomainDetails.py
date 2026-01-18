from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OnPremDomainDetails(_messages.Message):
    """OnPremDomainDetails is the message which contains details of on-prem
  domain which is trusted and needs to be migrated.

  Fields:
    disableSidFiltering: Optional. Option to disable SID filtering.
    domainName: Required. FQDN of the on-prem domain being migrated.
  """
    disableSidFiltering = _messages.BooleanField(1)
    domainName = _messages.StringField(2)