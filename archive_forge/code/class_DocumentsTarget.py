from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentsTarget(_messages.Message):
    """A target specified by a set of documents names.

  Fields:
    documents: The names of the documents to retrieve. In the format:
      `projects/{project_id}/databases/{database_id}/documents/{document_path}
      `. The request will fail if any of the document is not a child resource
      of the given `database`. Duplicate names will be elided.
  """
    documents = _messages.StringField(1, repeated=True)