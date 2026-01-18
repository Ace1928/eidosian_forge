from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
def _convert_structured_search_response(self, results: Sequence[SearchResult]) -> List[Document]:
    """Converts a sequence of search results to a list of LangChain documents."""
    import json
    from google.protobuf.json_format import MessageToDict
    documents: List[Document] = []
    for result in results:
        document_dict = MessageToDict(result.document._pb, preserving_proto_field_name=True)
        documents.append(Document(page_content=json.dumps(document_dict.get('struct_data', {})), metadata={'id': document_dict['id'], 'name': document_dict['name']}))
    return documents