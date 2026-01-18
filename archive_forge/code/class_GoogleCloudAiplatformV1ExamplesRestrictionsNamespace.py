from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ExamplesRestrictionsNamespace(_messages.Message):
    """Restrictions namespace for example-based explanations overrides.

  Fields:
    allow: The list of allowed tags.
    deny: The list of deny tags.
    namespaceName: The namespace name.
  """
    allow = _messages.StringField(1, repeated=True)
    deny = _messages.StringField(2, repeated=True)
    namespaceName = _messages.StringField(3)