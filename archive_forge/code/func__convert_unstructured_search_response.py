from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
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