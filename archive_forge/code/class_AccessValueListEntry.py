from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessValueListEntry(_messages.Message):
    """A AccessValueListEntry object.

    Fields:
      domain: [Pick one] A domain to grant access to. Any users signed in with
        the domain specified will be granted the specified access. Example:
        "example.com".
      groupByEmail: [Pick one] An email address of a Google Group to grant
        access to.
      role: [Required] Describes the rights granted to the user specified by
        the other member of the access object. The following string values are
        supported: READER, WRITER, OWNER.
      specialGroup: [Pick one] A special group to grant access to. Possible
        values include: projectOwners: Owners of the enclosing project.
        projectReaders: Readers of the enclosing project. projectWriters:
        Writers of the enclosing project. allAuthenticatedUsers: All
        authenticated BigQuery users.
      userByEmail: [Pick one] An email address of a user to grant access to.
        For example: fred@example.com.
      view: [Pick one] A view from a different dataset to grant access to.
        Queries executed against that view will have read access to tables in
        this dataset. The role field is not required when this field is set.
        If that view is updated by any user, access to the view needs to be
        granted again via an update operation.
    """
    domain = _messages.StringField(1)
    groupByEmail = _messages.StringField(2)
    role = _messages.StringField(3)
    specialGroup = _messages.StringField(4)
    userByEmail = _messages.StringField(5)
    view = _messages.MessageField('TableReference', 6)