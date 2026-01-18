from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import get_from_dict_or_env
from langchain_community.utilities.vertexai import get_client_info
def _create_search_request(self, query: str) -> SearchRequest:
    """Prepares a SearchRequest object."""
    from google.cloud.discoveryengine_v1beta import SearchRequest
    query_expansion_spec = SearchRequest.QueryExpansionSpec(condition=self.query_expansion_condition)
    spell_correction_spec = SearchRequest.SpellCorrectionSpec(mode=self.spell_correction_mode)
    if self.engine_data_type == 0:
        if self.get_extractive_answers:
            extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_answer_count=self.max_extractive_answer_count)
        else:
            extractive_content_spec = SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_segment_count=self.max_extractive_segment_count)
        content_search_spec = SearchRequest.ContentSearchSpec(extractive_content_spec=extractive_content_spec)
    elif self.engine_data_type == 1:
        content_search_spec = None
    elif self.engine_data_type in (2, 3):
        content_search_spec = SearchRequest.ContentSearchSpec(extractive_content_spec=SearchRequest.ContentSearchSpec.ExtractiveContentSpec(max_extractive_answer_count=self.max_extractive_answer_count), snippet_spec=SearchRequest.ContentSearchSpec.SnippetSpec(return_snippet=True))
    else:
        raise NotImplementedError('Only data store type 0 (Unstructured), 1 (Structured),2 (Website), or 3 (Blended) are supported currently.' + f' Got {self.engine_data_type}')
    return SearchRequest(query=query, filter=self.filter, serving_config=self._serving_config, page_size=self.max_documents, content_search_spec=content_search_spec, query_expansion_spec=query_expansion_spec, spell_correction_spec=spell_correction_spec)