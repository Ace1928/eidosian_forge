from __future__ import annotations
from typing import Any, Callable, List, Optional, Protocol, Tuple
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
def _collapse(self, docs: List[Document], token_max: Optional[int]=None, callbacks: Callbacks=None, **kwargs: Any) -> Tuple[List[Document], dict]:
    result_docs = docs
    length_func = self.combine_documents_chain.prompt_length
    num_tokens = length_func(result_docs, **kwargs)

    def _collapse_docs_func(docs: List[Document], **kwargs: Any) -> str:
        return self._collapse_chain.run(input_documents=docs, callbacks=callbacks, **kwargs)
    _token_max = token_max or self.token_max
    retries: int = 0
    while num_tokens is not None and num_tokens > _token_max:
        new_result_doc_list = split_list_of_docs(result_docs, length_func, _token_max, **kwargs)
        result_docs = []
        for docs in new_result_doc_list:
            new_doc = collapse_docs(docs, _collapse_docs_func, **kwargs)
            result_docs.append(new_doc)
        num_tokens = length_func(result_docs, **kwargs)
        retries += 1
        if self.collapse_max_retries and retries == self.collapse_max_retries:
            raise ValueError(f'Exceed {self.collapse_max_retries} tries to                         collapse document to {_token_max} tokens.')
    return (result_docs, {})