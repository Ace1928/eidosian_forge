from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra
def _embed_text(self, texts: List[str]) -> List[List[float]]:
    inputs = self.transformer_tokenizer(texts, max_length=self.max_seq_len, truncation=True, padding=self.padding, return_tensors='pt')
    return self._embed(inputs).tolist()