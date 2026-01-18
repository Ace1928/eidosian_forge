from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class PyMuPDFParser(BaseBlobParser):
    """Parse `PDF` using `PyMuPDF`."""

    def __init__(self, text_kwargs: Optional[Mapping[str, Any]]=None, extract_images: bool=False) -> None:
        """Initialize the parser.

        Args:
            text_kwargs: Keyword arguments to pass to ``fitz.Page.get_text()``.
        """
        self.text_kwargs = text_kwargs or {}
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import fitz
        with blob.as_bytes_io() as file_path:
            if blob.data is None:
                doc = fitz.open(file_path)
            else:
                doc = fitz.open(stream=file_path, filetype='pdf')
            yield from [Document(page_content=page.get_text(**self.text_kwargs) + self._extract_images_from_page(doc, page), metadata=dict({'source': blob.source, 'file_path': blob.source, 'page': page.number, 'total_pages': len(doc)}, **{k: doc.metadata[k] for k in doc.metadata if type(doc.metadata[k]) in [str, int]})) for page in doc]

    def _extract_images_from_page(self, doc: fitz.fitz.Document, page: fitz.fitz.Page) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images:
            return ''
        import fitz
        img_list = page.get_images()
        imgs = []
        for img in img_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            imgs.append(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1))
        return extract_from_images_with_rapidocr(imgs)