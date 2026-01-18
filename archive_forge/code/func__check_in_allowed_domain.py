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
def _check_in_allowed_domain(url: str, limit_to_domains: Sequence[str]) -> bool:
    """Check if a URL is in the allowed domains.

    Args:
        url (str): The input URL.
        limit_to_domains (Sequence[str]): The allowed domains.

    Returns:
        bool: True if the URL is in the allowed domains, False otherwise.
    """
    scheme, domain = _extract_scheme_and_domain(url)
    for allowed_domain in limit_to_domains:
        allowed_scheme, allowed_domain = _extract_scheme_and_domain(allowed_domain)
        if scheme == allowed_scheme and domain == allowed_domain:
            return True
    return False