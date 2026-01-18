from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.chat_models import (
from langchain_core.messages import (
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_community.llms.cohere import BaseCohere
def _get_generation_info(self, response: Any) -> Dict[str, Any]:
    """Get the generation info from cohere API response."""
    return {'documents': response.documents, 'citations': response.citations, 'search_results': response.search_results, 'search_queries': response.search_queries, 'token_count': response.token_count}