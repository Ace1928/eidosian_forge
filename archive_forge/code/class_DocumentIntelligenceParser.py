from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class DocumentIntelligenceParser(BaseBlobParser):
    """Loads a PDF with Azure Document Intelligence
    (formerly Form Recognizer) and chunks at character level."""

    def __init__(self, client: Any, model: str):
        warnings.warn('langchain_community.document_loaders.parsers.pdf.DocumentIntelligenceParserand langchain_community.document_loaders.pdf.DocumentIntelligenceLoader are deprecated. Please upgrade to langchain_community.document_loaders.DocumentIntelligenceLoader for any file parsing purpose using Azure Document Intelligence service.')
        self.client = client
        self.model = model

    def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:
        for p in result.pages:
            content = ' '.join([line.content for line in p.lines])
            d = Document(page_content=content, metadata={'source': blob.source, 'page': p.page_number})
            yield d

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        with blob.as_bytes_io() as file_obj:
            poller = self.client.begin_analyze_document(self.model, file_obj)
            result = poller.result()
            docs = self._generate_docs(blob, result)
            yield from docs