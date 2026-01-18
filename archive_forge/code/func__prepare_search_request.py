from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
def _prepare_search_request(self, query: str, **kwargs: Any) -> 'SearchDocumentsRequest':
    from google.cloud.contentwarehouse_v1 import DocumentQuery, SearchDocumentsRequest
    try:
        user_ldap = kwargs['user_ldap']
    except KeyError:
        raise ValueError('Argument user_ldap should be provided!')
    request_metadata = self._prepare_request_metadata(user_ldap=user_ldap)
    schemas = []
    if self.schema_id:
        schemas.append(self.client.document_schema_path(project=self.project_number, location=self.location, document_schema=self.schema_id))
    return SearchDocumentsRequest(parent=self.client.common_location_path(self.project_number, self.location), request_metadata=request_metadata, document_query=DocumentQuery(query=query, is_nl_query=True, document_schema_names=schemas), qa_size_limit=self.qa_size_limit)