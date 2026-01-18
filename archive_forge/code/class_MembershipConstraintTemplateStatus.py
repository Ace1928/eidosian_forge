from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MembershipConstraintTemplateStatus(_messages.Message):
    """MembershipConstratinTemplateStatus contains status information, e.g.
  whether the template has been created on the member cluster.

  Fields:
    created: status.created from the constraint template.
  """
    created = _messages.BooleanField(1)