from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckTransitiveMembershipResponse(_messages.Message):
    """The response message for MembershipsService.CheckTransitiveMembership.

  Fields:
    hasMembership: Response does not include the possible roles of a member
      since the behavior of this rpc is not all-or-nothing unlike the other
      rpcs. So, it may not be possible to list all the roles definitively, due
      to possible lack of authorization in some of the paths.
  """
    hasMembership = _messages.BooleanField(1)