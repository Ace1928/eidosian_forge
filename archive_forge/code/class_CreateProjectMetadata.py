from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateProjectMetadata(_messages.Message):
    """A status object which is used as the `metadata` field for the Operation
  returned by CreateProject. It provides insight for when significant phases
  of Project creation have completed.

  Fields:
    createTime: Creation time of the project creation workflow.
    gettable: True if the project can be retrieved using `GetProject`. No
      other operations on the project are guaranteed to work until the project
      creation is complete.
    ready: True if the project creation process is complete.
  """
    createTime = _messages.StringField(1)
    gettable = _messages.BooleanField(2)
    ready = _messages.BooleanField(3)