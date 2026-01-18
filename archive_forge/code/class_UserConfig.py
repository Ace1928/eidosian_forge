from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserConfig(_messages.Message):
    """Defines where users exist in the LDAP directory.

  Fields:
    baseDn: Required. The location of the subtree in the LDAP directory to
      search for user entries.
    filter: Optional. Filter to apply when searching for the user. This can be
      used to further restrict the user accounts which are allowed to login.
      This defaults to "(objectClass=User)".
    idAttribute: Optional. Determines which attribute to use as the user's
      identity after they are authenticated. This is distinct from the
      loginAttribute field to allow users to login with a username, but then
      have their actual identifier be an email address or full Distinguished
      Name (DN). For example, setting loginAttribute to "sAMAccountName" and
      identifierAttribute to "userPrincipalName" would allow a user to login
      as "bsmith", but actual RBAC policies for the user would be written as
      "bsmith@example.com". Using "userPrincipalName" is recommended since
      this will be unique for each user. This defaults to "userPrincipalName".
    loginAttribute: Optional. The name of the attribute which matches against
      the input username. This is used to find the user in the LDAP database
      e.g. "(=)" and is combined with the optional filter field. This defaults
      to "userPrincipalName".
  """
    baseDn = _messages.StringField(1)
    filter = _messages.StringField(2)
    idAttribute = _messages.StringField(3)
    loginAttribute = _messages.StringField(4)