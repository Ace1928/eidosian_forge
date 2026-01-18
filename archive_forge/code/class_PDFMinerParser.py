from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class PDFMinerParser(BaseBlobParser):
    """Parse `PDF` using `PDFMiner`."""

    def __init__(self, extract_images: bool=False, *, concatenate_pages: bool=True):
        """Initialize a parser based on PDFMiner.

        Args:
            extract_images: Whether to extract images from PDF.
            concatenate_pages: If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        self.extract_images = extract_images
        self.concatenate_pages = concatenate_pages

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""
        if not self.extract_images:
            from pdfminer.high_level import extract_text
            with blob.as_bytes_io() as pdf_file_obj:
                if self.concatenate_pages:
                    text = extract_text(pdf_file_obj)
                    metadata = {'source': blob.source}
                    yield Document(page_content=text, metadata=metadata)
                else:
                    from pdfminer.pdfpage import PDFPage
                    pages = PDFPage.get_pages(pdf_file_obj)
                    for i, _ in enumerate(pages):
                        text = extract_text(pdf_file_obj, page_numbers=[i])
                        metadata = {'source': blob.source, 'page': str(i)}
                        yield Document(page_content=text, metadata=metadata)
        else:
            import io
            from pdfminer.converter import PDFPageAggregator, TextConverter
            from pdfminer.layout import LAParams
            from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
            from pdfminer.pdfpage import PDFPage
            text_io = io.StringIO()
            with blob.as_bytes_io() as pdf_file_obj:
                pages = PDFPage.get_pages(pdf_file_obj)
                rsrcmgr = PDFResourceManager()
                device_for_text = TextConverter(rsrcmgr, text_io, laparams=LAParams())
                device_for_image = PDFPageAggregator(rsrcmgr, laparams=LAParams())
                interpreter_for_text = PDFPageInterpreter(rsrcmgr, device_for_text)
                interpreter_for_image = PDFPageInterpreter(rsrcmgr, device_for_image)
                for i, page in enumerate(pages):
                    interpreter_for_text.process_page(page)
                    interpreter_for_image.process_page(page)
                    content = text_io.getvalue() + self._extract_images_from_page(device_for_image.get_result())
                    text_io.truncate(0)
                    text_io.seek(0)
                    metadata = {'source': blob.source, 'page': str(i)}
                    yield Document(page_content=content, metadata=metadata)

    def _extract_images_from_page(self, page: pdfminer.layout.LTPage) -> str:
        """Extract images from page and get the text with RapidOCR."""
        import pdfminer

        def get_image(layout_object: Any) -> Any:
            if isinstance(layout_object, pdfminer.layout.LTImage):
                return layout_object
            if isinstance(layout_object, pdfminer.layout.LTContainer):
                for child in layout_object:
                    return get_image(child)
            else:
                return None
        images = []
        for img in list(filter(bool, map(get_image, page))):
            if img.stream['Filter'].name in _PDF_FILTER_WITHOUT_LOSS:
                images.append(np.frombuffer(img.stream.get_data(), dtype=np.uint8).reshape(img.stream['Height'], img.stream['Width'], -1))
            elif img.stream['Filter'].name in _PDF_FILTER_WITH_LOSS:
                images.append(img.stream.get_data())
            else:
                warnings.warn('Unknown PDF Filter!')
        return extract_from_images_with_rapidocr(images)