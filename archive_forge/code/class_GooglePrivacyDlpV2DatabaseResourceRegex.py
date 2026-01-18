from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2DatabaseResourceRegex(_messages.Message):
    """A pattern to match against one or more database resources. At least one
  pattern must be specified. Regular expressions use RE2
  [syntax](https://github.com/google/re2/wiki/Syntax); a guide can be found
  under the google/re2 repository on GitHub.

  Fields:
    databaseRegex: Regex to test the database name against. If empty, all
      databases match.
    databaseResourceNameRegex: Regex to test the database resource's name
      against. An example of a database resource name is a table's name. Other
      database resource names like view names could be included in the future.
      If empty, all database resources match.
    instanceRegex: Regex to test the instance name against. If empty, all
      instances match.
    projectIdRegex: For organizations, if unset, will match all projects. Has
      no effect for Data Profile configurations created within a project.
  """
    databaseRegex = _messages.StringField(1)
    databaseResourceNameRegex = _messages.StringField(2)
    instanceRegex = _messages.StringField(3)
    projectIdRegex = _messages.StringField(4)