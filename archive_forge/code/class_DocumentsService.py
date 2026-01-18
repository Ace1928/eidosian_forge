from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
class DocumentsService(base_api.BaseApiService):
    """Service class for the documents resource."""
    _NAME = 'documents'

    def __init__(self, client):
        super(LanguageV1.DocumentsService, self).__init__(client)
        self._upload_configs = {}

    def AnalyzeEntities(self, request, global_params=None):
        """Finds named entities (currently proper names and common nouns) in the text along with entity types, salience, mentions for each entity, and other properties.

      Args:
        request: (AnalyzeEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeEntitiesResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeEntities')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeEntities.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.analyzeEntities', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:analyzeEntities', request_field='<request>', request_type_name='AnalyzeEntitiesRequest', response_type_name='AnalyzeEntitiesResponse', supports_download=False)

    def AnalyzeEntitySentiment(self, request, global_params=None):
        """Finds entities, similar to AnalyzeEntities in the text and analyzes sentiment associated with each entity and its mentions.

      Args:
        request: (AnalyzeEntitySentimentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeEntitySentimentResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeEntitySentiment')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeEntitySentiment.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.analyzeEntitySentiment', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:analyzeEntitySentiment', request_field='<request>', request_type_name='AnalyzeEntitySentimentRequest', response_type_name='AnalyzeEntitySentimentResponse', supports_download=False)

    def AnalyzeSentiment(self, request, global_params=None):
        """Analyzes the sentiment of the provided text.

      Args:
        request: (AnalyzeSentimentRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeSentimentResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeSentiment')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeSentiment.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.analyzeSentiment', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:analyzeSentiment', request_field='<request>', request_type_name='AnalyzeSentimentRequest', response_type_name='AnalyzeSentimentResponse', supports_download=False)

    def AnalyzeSyntax(self, request, global_params=None):
        """Analyzes the syntax of the text and provides sentence boundaries and tokenization along with part of speech tags, dependency trees, and other properties.

      Args:
        request: (AnalyzeSyntaxRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeSyntaxResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeSyntax')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeSyntax.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.analyzeSyntax', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:analyzeSyntax', request_field='<request>', request_type_name='AnalyzeSyntaxRequest', response_type_name='AnalyzeSyntaxResponse', supports_download=False)

    def AnnotateText(self, request, global_params=None):
        """A convenience method that provides all the features that analyzeSentiment, analyzeEntities, and analyzeSyntax provide in one call.

      Args:
        request: (AnnotateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnnotateTextResponse) The response message.
      """
        config = self.GetMethodConfig('AnnotateText')
        return self._RunMethod(config, request, global_params=global_params)
    AnnotateText.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.annotateText', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:annotateText', request_field='<request>', request_type_name='AnnotateTextRequest', response_type_name='AnnotateTextResponse', supports_download=False)

    def ClassifyText(self, request, global_params=None):
        """Classifies a document into categories.

      Args:
        request: (ClassifyTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ClassifyTextResponse) The response message.
      """
        config = self.GetMethodConfig('ClassifyText')
        return self._RunMethod(config, request, global_params=global_params)
    ClassifyText.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.classifyText', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:classifyText', request_field='<request>', request_type_name='ClassifyTextRequest', response_type_name='ClassifyTextResponse', supports_download=False)

    def ModerateText(self, request, global_params=None):
        """Moderates a document for harmful and sensitive categories.

      Args:
        request: (ModerateTextRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModerateTextResponse) The response message.
      """
        config = self.GetMethodConfig('ModerateText')
        return self._RunMethod(config, request, global_params=global_params)
    ModerateText.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='language.documents.moderateText', ordered_params=[], path_params=[], query_params=[], relative_path='v1/documents:moderateText', request_field='<request>', request_type_name='ModerateTextRequest', response_type_name='ModerateTextResponse', supports_download=False)