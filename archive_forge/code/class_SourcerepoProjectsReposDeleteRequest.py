from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposDeleteRequest(_messages.Message):
    """A SourcerepoProjectsReposDeleteRequest object.

  Fields:
    name: The name of the repo to delete. Values are of the form
      `projects//repos/`.
  """
    name = _messages.StringField(1, required=True)