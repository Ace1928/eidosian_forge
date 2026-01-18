from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchRangeRequest(_messages.Message):
    """Request to search for an unused range within allocated ranges.

  Fields:
    ipPrefixLength: Required. The prefix length of the IP range. Use usual
      CIDR range notation. For example, '30' to find unused x.x.x.x/30 CIDR
      range. Actual range will be determined using allocated range for the
      consumer peered network and returned in the result.
    network: Network name in the consumer project. This network must have been
      already peered with a shared VPC network using CreateConnection method.
      Must be in a form 'projects/{project}/global/networks/{network}'.
      {project} is a project number, as in '12345' {network} is network name.
  """
    ipPrefixLength = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    network = _messages.StringField(2)