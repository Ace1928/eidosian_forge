from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CommitTemplateVersionRequest(_messages.Message):
    """Commit will add a new TemplateVersion to an existing template.

  Fields:
    templateVersion: TemplateVersion object to create.
  """
    templateVersion = _messages.MessageField('TemplateVersion', 1)