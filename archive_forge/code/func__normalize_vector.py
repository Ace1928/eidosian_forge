import asyncio
import json
import os
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.runnables.config import run_in_executor
def _normalize_vector(self, embeddings: List[float]) -> List[float]:
    """Normalize the embedding to a unit vector."""
    emb = np.array(embeddings)
    norm_emb = emb / np.linalg.norm(emb)
    return norm_emb.tolist()