from enum import Enum
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field, root_validator
from langchain.callbacks.manager import (
from langchain.chains.base import Chain
from langchain.evaluation.schema import PairwiseStringEvaluator, StringEvaluator
from langchain.schema import RUN_KEY
from langchain.utils.math import cosine_similarity
@root_validator(pre=False)
def _validate_tiktoken_installed(cls, values: Dict[str, Any]) -> Dict[str, Any]:
    """Validate that the TikTok library is installed.

        Args:
            values (Dict[str, Any]): The values to validate.

        Returns:
            Dict[str, Any]: The validated values.
        """
    embeddings = values.get('embeddings')
    if isinstance(embeddings, OpenAIEmbeddings):
        try:
            import tiktoken
        except ImportError:
            raise ImportError('The tiktoken library is required to use the default OpenAI embeddings with embedding distance evaluators. Please either manually select a different Embeddings object or install tiktoken using `pip install tiktoken`.')
    return values