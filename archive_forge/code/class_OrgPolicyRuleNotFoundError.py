from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
class OrgPolicyRuleNotFoundError(OrgPolicyError):
    """Exception for a nonexistent rule on an organization policy."""