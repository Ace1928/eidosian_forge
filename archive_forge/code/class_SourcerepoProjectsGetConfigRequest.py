from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsGetConfigRequest(_messages.Message):
    """A SourcerepoProjectsGetConfigRequest object.

  Fields:
    name: The name of the requested project. Values are of the form
      `projects/`.
  """
    name = _messages.StringField(1, required=True)