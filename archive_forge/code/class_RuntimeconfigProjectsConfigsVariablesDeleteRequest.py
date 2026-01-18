from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeconfigProjectsConfigsVariablesDeleteRequest(_messages.Message):
    """A RuntimeconfigProjectsConfigsVariablesDeleteRequest object.

  Fields:
    name: The name of the variable to delete, in the format:
      `projects/[PROJECT_ID]/configs/[CONFIG_NAME]/variables/[VARIABLE_NAME]`
    recursive: Set to `true` to recursively delete multiple variables with the
      same prefix.
  """
    name = _messages.StringField(1, required=True)
    recursive = _messages.BooleanField(2)