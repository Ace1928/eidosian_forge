from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAccessList(_messages.Message):
    """List of users and groups that are granted access to a service or
  visibility label.

  Fields:
    members: Members that are granted access.  - "user:{$user_email}" - Grant
      access to an individual user - "group:{$group_email}" - Grant access to
      direct members of the group - "domain:{$domain}" - Grant access to all
      members of the domain. For now,      domain membership check will be
      similar to Devconsole/TT check:      compare domain part of the user
      email to configured domain name.      When IAM integration is complete,
      this will be replaced with IAM      check.
  """
    members = _messages.StringField(1, repeated=True)