from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IdentityServiceGroupConfig(_messages.Message):
    """Contains the properties for locating and authenticating groups in the
  directory.

  Fields:
    baseDn: Required. The location of the subtree in the LDAP directory to
      search for group entries.
    filter: Optional. Optional filter to be used when searching for groups a
      user belongs to. This can be used to explicitly match only certain
      groups in order to reduce the amount of groups returned for each user.
      This defaults to "(objectClass=Group)".
    idAttribute: Optional. The identifying name of each group a user belongs
      to. For example, if this is set to "distinguishedName" then RBACs and
      other group expectations should be written as full DNs. This defaults to
      "distinguishedName".
  """
    baseDn = _messages.StringField(1)
    filter = _messages.StringField(2)
    idAttribute = _messages.StringField(3)