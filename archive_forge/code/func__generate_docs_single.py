import logging
from typing import Any, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def _generate_docs_single(self, result: Any) -> Iterator[Document]:
    yield Document(page_content=result.content, metadata={})