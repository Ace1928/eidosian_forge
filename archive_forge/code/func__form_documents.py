from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
def _form_documents(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> List[Document]:
    """Format context from this conversation to buffer."""
    exclude = set(self.exclude_input_keys)
    exclude.add(self.memory_key)
    filtered_inputs = {k: v for k, v in inputs.items() if k not in exclude}
    texts = [f'{k}: {v}' for k, v in list(filtered_inputs.items()) + list(outputs.items())]
    page_content = '\n'.join(texts)
    return [Document(page_content=page_content)]