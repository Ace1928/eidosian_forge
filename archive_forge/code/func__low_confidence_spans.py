from __future__ import annotations
import re
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_community.llms.openai import OpenAI
from langchain_core.callbacks import (
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import Generation
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever
from langchain.chains.base import Chain
from langchain.chains.flare.prompts import (
from langchain.chains.llm import LLMChain
def _low_confidence_spans(tokens: Sequence[str], log_probs: Sequence[float], min_prob: float, min_token_gap: int, num_pad_tokens: int) -> List[str]:
    _low_idx = np.where(np.exp(log_probs) < min_prob)[0]
    low_idx = [i for i in _low_idx if re.search('\\w', tokens[i])]
    if len(low_idx) == 0:
        return []
    spans = [[low_idx[0], low_idx[0] + num_pad_tokens + 1]]
    for i, idx in enumerate(low_idx[1:]):
        end = idx + num_pad_tokens + 1
        if idx - low_idx[i] < min_token_gap:
            spans[-1][1] = end
        else:
            spans.append([idx, end])
    return [''.join(tokens[start:end]) for start, end in spans]