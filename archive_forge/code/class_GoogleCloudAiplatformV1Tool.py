from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1Tool(_messages.Message):
    """Tool details that the model may use to generate response. A `Tool` is a
  piece of code that enables the system to interact with external systems to
  perform an action, or set of actions, outside of knowledge and scope of the
  model. A Tool object should contain exactly one type of Tool (e.g
  FunctionDeclaration, Retrieval or GoogleSearchRetrieval).

  Fields:
    functionDeclarations: Optional. Function tool type. One or more function
      declarations to be passed to the model along with the current user
      query. Model may decide to call a subset of these functions by
      populating FunctionCall in the response. User should provide a
      FunctionResponse for each function call in the next turn. Based on the
      function responses, Model will generate the final response back to the
      user. Maximum 64 function declarations can be provided.
    googleSearchRetrieval: Optional. GoogleSearchRetrieval tool type.
      Specialized retrieval tool that is powered by Google search.
    retrieval: Optional. Retrieval tool type. System will always execute the
      provided retrieval tool(s) to get external knowledge to answer the
      prompt. Retrieval results are presented to the model for generation.
  """
    functionDeclarations = _messages.MessageField('GoogleCloudAiplatformV1FunctionDeclaration', 1, repeated=True)
    googleSearchRetrieval = _messages.MessageField('GoogleCloudAiplatformV1GoogleSearchRetrieval', 2)
    retrieval = _messages.MessageField('GoogleCloudAiplatformV1Retrieval', 3)