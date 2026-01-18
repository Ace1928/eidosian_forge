from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
@deprecated(since='0.0.33', removal='0.2.0', alternative_import='langchain_google_community.VertexAIMultiTurnSearchRetriever')
class GoogleVertexAIMultiTurnSearchRetriever(BaseRetriever, _BaseGoogleVertexAISearchRetriever):
    """`Google Vertex AI Search` retriever for multi-turn conversations."""
    conversation_id: str = '-'
    'Vertex AI Search Conversation ID.'
    _client: ConversationalSearchServiceClient
    _serving_config: str

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.ignore
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        from google.cloud.discoveryengine_v1beta import ConversationalSearchServiceClient
        self._client = ConversationalSearchServiceClient(credentials=self.credentials, client_options=self.client_options, client_info=get_client_info(module='vertex-ai-search'))
        if not self.data_store_id:
            raise ValueError('data_store_id is required for MultiTurnSearchRetriever.')
        self._serving_config = self._client.serving_config_path(project=self.project_id, location=self.location_id, data_store=self.data_store_id, serving_config=self.serving_config_id)
        if self.engine_data_type == 1 or self.engine_data_type == 3:
            raise NotImplementedError('Data store type 1 (Structured) and 3 (Blended)is not currently supported for multi-turn search.' + f' Got {self.engine_data_type}')

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get documents relevant for a query."""
        from google.cloud.discoveryengine_v1beta import ConverseConversationRequest, TextInput
        request = ConverseConversationRequest(name=self._client.conversation_path(self.project_id, self.location_id, self.data_store_id, self.conversation_id), serving_config=self._serving_config, query=TextInput(input=query))
        response = self._client.converse_conversation(request)
        if self.engine_data_type == 2:
            return self._convert_website_search_response(response.search_results, 'extractive_answers')
        return self._convert_unstructured_search_response(response.search_results, 'extractive_answers')