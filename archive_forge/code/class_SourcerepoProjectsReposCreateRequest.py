from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourcerepoProjectsReposCreateRequest(_messages.Message):
    """A SourcerepoProjectsReposCreateRequest object.

  Fields:
    parent: The project in which to create the repo. Values are of the form
      `projects/`.
    repo: A Repo resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    repo = _messages.MessageField('Repo', 2)