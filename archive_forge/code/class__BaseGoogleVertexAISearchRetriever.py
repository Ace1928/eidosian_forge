from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
class _BaseGoogleVertexAISearchRetriever(BaseModel):
    project_id: str
    'Google Cloud Project ID.'
    data_store_id: Optional[str] = None
    'Vertex AI Search data store ID.'
    search_engine_id: Optional[str] = None
    'Vertex AI Search app ID.'
    location_id: str = 'global'
    'Vertex AI Search data store location.'
    serving_config_id: str = 'default_config'
    'Vertex AI Search serving config ID.'
    credentials: Any = None
    'The default custom credentials (google.auth.credentials.Credentials) to use\n    when making API calls. If not provided, credentials will be ascertained from\n    the environment.'
    engine_data_type: int = Field(default=0, ge=0, le=3)
    ' Defines the Vertex AI Search app data type\n    0 - Unstructured data \n    1 - Structured data\n    2 - Website data\n    3 - Blended search\n    '

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validates the environment."""
        try:
            from google.cloud import discoveryengine_v1beta
        except ImportError as exc:
            raise ImportError('google.cloud.discoveryengine is not installed.Please install it with pip install google-cloud-discoveryengine>=0.11.10') from exc
        try:
            from google.api_core.exceptions import InvalidArgument
        except ImportError as exc:
            raise ImportError('google.api_core.exceptions is not installed. Please install it with pip install google-api-core') from exc
        values['project_id'] = get_from_dict_or_env(values, 'project_id', 'PROJECT_ID')
        try:
            values['data_store_id'] = get_from_dict_or_env(values, 'data_store_id', 'DATA_STORE_ID')
            values['search_engine_id'] = get_from_dict_or_env(values, 'search_engine_id', 'SEARCH_ENGINE_ID')
        except Exception:
            pass
        return values

    @property
    def client_options(self) -> 'ClientOptions':
        from google.api_core.client_options import ClientOptions
        return ClientOptions(api_endpoint=f'{self.location_id}-discoveryengine.googleapis.com' if self.location_id != 'global' else None)

    def _convert_structured_search_response(self, results: Sequence[SearchResult]) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        import json
        from google.protobuf.json_format import MessageToDict
        documents: List[Document] = []
        for result in results:
            document_dict = MessageToDict(result.document._pb, preserving_proto_field_name=True)
            documents.append(Document(page_content=json.dumps(document_dict.get('struct_data', {})), metadata={'id': document_dict['id'], 'name': document_dict['name']}))
        return documents

    def _convert_unstructured_search_response(self, results: Sequence[SearchResult], chunk_type: str) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        from google.protobuf.json_format import MessageToDict
        documents: List[Document] = []
        for result in results:
            document_dict = MessageToDict(result.document._pb, preserving_proto_field_name=True)
            derived_struct_data = document_dict.get('derived_struct_data')
            if not derived_struct_data:
                continue
            doc_metadata = document_dict.get('struct_data', {})
            doc_metadata['id'] = document_dict['id']
            if chunk_type not in derived_struct_data:
                continue
            for chunk in derived_struct_data[chunk_type]:
                chunk_metadata = doc_metadata.copy()
                chunk_metadata['source'] = derived_struct_data.get('link', '')
                if chunk_type == 'extractive_answers':
                    chunk_metadata['source'] += f':{chunk.get('pageNumber', '')}'
                documents.append(Document(page_content=chunk.get('content', ''), metadata=chunk_metadata))
        return documents

    def _convert_website_search_response(self, results: Sequence[SearchResult], chunk_type: str) -> List[Document]:
        """Converts a sequence of search results to a list of LangChain documents."""
        from google.protobuf.json_format import MessageToDict
        documents: List[Document] = []
        for result in results:
            document_dict = MessageToDict(result.document._pb, preserving_proto_field_name=True)
            derived_struct_data = document_dict.get('derived_struct_data')
            if not derived_struct_data:
                continue
            doc_metadata = document_dict.get('struct_data', {})
            doc_metadata['id'] = document_dict['id']
            doc_metadata['source'] = derived_struct_data.get('link', '')
            if chunk_type not in derived_struct_data:
                continue
            text_field = 'snippet' if chunk_type == 'snippets' else 'content'
            for chunk in derived_struct_data[chunk_type]:
                documents.append(Document(page_content=chunk.get(text_field, ''), metadata=doc_metadata))
        if not documents:
            print(f'No {chunk_type} could be found.')
            if chunk_type == 'extractive_answers':
                print('Make sure that your data store is using Advanced Website Indexing.\nhttps://cloud.google.com/generative-ai-app-builder/docs/about-advanced-features#advanced-website-indexing')
        return documents