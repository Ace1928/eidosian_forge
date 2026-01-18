from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory.chat_memory import BaseMemory
from langchain.memory.utils import get_prompt_input_key
def _documents_to_memory_variables(self, docs: List[Document]) -> Dict[str, Union[List[Document], str]]:
    result: Union[List[Document], str]
    if not self.return_docs:
        result = '\n'.join([doc.page_content for doc in docs])
    else:
        result = docs
    return {self.memory_key: result}