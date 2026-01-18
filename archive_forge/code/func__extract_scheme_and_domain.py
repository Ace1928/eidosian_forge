from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.chains.api.prompt import API_RESPONSE_PROMPT, API_URL_PROMPT
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
def _extract_scheme_and_domain(url: str) -> Tuple[str, str]:
    """Extract the scheme + domain from a given URL.

    Args:
        url (str): The input URL.

    Returns:
        return a 2-tuple of scheme and domain
    """
    parsed_uri = urlparse(url)
    return (parsed_uri.scheme, parsed_uri.netloc)