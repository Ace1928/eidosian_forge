from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class PDFPlumberParser(BaseBlobParser):
    """Parse `PDF` with `PDFPlumber`."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]]=None, dedupe: bool=False, extract_images: bool=False) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``pdfplumber.Page.extract_text()``
            dedupe: Avoiding the error of duplicate characters if `dedupe=True`.
        """
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pdfplumber
        with blob.as_bytes_io() as file_path:
            doc = pdfplumber.open(file_path)
            yield from [Document(page_content=self._process_page_content(page) + '\n' + self._extract_images_from_page(page), metadata=dict({'source': blob.source, 'file_path': blob.source, 'page': page.page_number - 1, 'total_pages': len(doc.pages)}, **{k: doc.metadata[k] for k in doc.metadata if type(doc.metadata[k]) in [str, int]})) for page in doc.pages]

    def _process_page_content(self, page: pdfplumber.page.Page) -> str:
        """Process the page content based on dedupe."""
        if self.dedupe:
            return page.dedupe_chars().extract_text(**self.text_kwargs)
        return page.extract_text(**self.text_kwargs)

    def _extract_images_from_page(self, page: pdfplumber.page.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ''
        images = []
        for img in page.images:
            if img['stream']['Filter'].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(np.frombuffer(img['stream'].get_data(), dtype=np.uint8).reshape(img['stream']['Height'], img['stream']['Width'], -1))
            elif img['stream']['Filter'].name in _PDF_FILTER_WITH_LOSS:
                images.append(img['stream'].get_data())
            else:
                warnings.warn('Unknown PDF Filter!')
        return extract_from_images_with_rapidocr(images)