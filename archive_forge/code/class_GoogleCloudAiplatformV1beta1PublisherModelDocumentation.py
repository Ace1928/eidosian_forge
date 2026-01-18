from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1PublisherModelDocumentation(_messages.Message):
    """A named piece of documentation.

  Fields:
    content: Required. Content of this piece of document (in Markdown format).
    title: Required. E.g., OVERVIEW, USE CASES, DOCUMENTATION, SDK & SAMPLES,
      JAVA, NODE.JS, etc..
  """
    content = _messages.StringField(1)
    title = _messages.StringField(2)