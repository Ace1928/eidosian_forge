from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GerritSourceContext(_messages.Message):
    """A SourceContext referring to a Gerrit project.

  Fields:
    aliasContext: An alias, which may be a branch or tag.
    gerritProject: The full project name within the host. Projects may be
      nested, so "project/subproject" is a valid project name. The "repo name"
      is the hostURI/project.
    hostUri: The URI of a running Gerrit instance.
    revisionId: A revision (commit) ID.
  """
    aliasContext = _messages.MessageField('AliasContext', 1)
    gerritProject = _messages.StringField(2)
    hostUri = _messages.StringField(3)
    revisionId = _messages.StringField(4)