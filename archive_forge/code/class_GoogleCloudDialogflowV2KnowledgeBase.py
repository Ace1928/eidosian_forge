from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2KnowledgeBase(_messages.Message):
    """A knowledge base represents a collection of knowledge documents that you
  provide to Dialogflow. Your knowledge documents contain information that may
  be useful during conversations with end-users. Some Dialogflow features use
  knowledge bases when looking for a response to an end-user input. For more
  information, see the [knowledge base
  guide](https://cloud.google.com/dialogflow/docs/how/knowledge-bases). Note:
  The `projects.agent.knowledgeBases` resource is deprecated; only use
  `projects.knowledgeBases`.

  Fields:
    displayName: Required. The display name of the knowledge base. The name
      must be 1024 bytes or less; otherwise, the creation request fails.
    languageCode: Language which represents the KnowledgeBase. When the
      KnowledgeBase is created/updated, expect this to be present for non en-
      us languages. When unspecified, the default language code en-us applies.
    name: The knowledge base resource name. The name must be empty when
      creating a knowledge base. Format:
      `projects//locations//knowledgeBases/`.
  """
    displayName = _messages.StringField(1)
    languageCode = _messages.StringField(2)
    name = _messages.StringField(3)