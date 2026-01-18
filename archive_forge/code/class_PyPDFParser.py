from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class PyPDFParser(BaseBlobParser):
    """Load `PDF` using `pypdf`"""

    def __init__(self, password: Optional[Union[str, bytes]]=None, extract_images: bool=False):
        self.password = password
        self.extract_images = extract_images

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        import pypdf
        with blob.as_bytes_io() as pdf_file_obj:
            pdf_reader = pypdf.PdfReader(pdf_file_obj, password=self.password)
            yield from [Document(page_content=page.extract_text() + self._extract_images_from_page(page), metadata={'source': blob.source, 'page': page_number}) for page_number, page in enumerate(pdf_reader.pages)]

    def _extract_images_from_page(self, page: pypdf._page.PageObject) -> str:
        """Extract images from page and get the text with RapidOCR."""
        if not self.extract_images or '/XObject' not in page['/Resources'].keys():
            return ''
        xObject = page['/Resources']['/XObject'].get_object()
        images = []
        for obj in xObject:
            if xObject[obj]['/Subtype'] == '/Image':
                if xObject[obj]['/Filter'][1:] in _PDF_FILTER_WITHOUT_LOSS:
                    height, width = (xObject[obj]['/Height'], xObject[obj]['/Width'])
                    images.append(np.frombuffer(xObject[obj].get_data(), dtype=np.uint8).reshape(height, width, -1))
                elif xObject[obj]['/Filter'][1:] in _PDF_FILTER_WITH_LOSS:
                    images.append(xObject[obj].get_data())
                else:
                    warnings.warn('Unknown PDF Filter!')
        return extract_from_images_with_rapidocr(images)