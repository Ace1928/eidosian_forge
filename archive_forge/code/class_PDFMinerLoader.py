import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class PDFMinerLoader(BasePDFLoader):
    """Load `PDF` files using `PDFMiner`."""

    def __init__(self, file_path: str, *, headers: Optional[Dict]=None, extract_images: bool=False, concatenate_pages: bool=True) -> None:
        """Initialize with file path.

        Args:
            extract_images: Whether to extract images from PDF.
            concatenate_pages: If True, concatenate all PDF pages into one a single
                               document. Otherwise, return one document per page.
        """
        try:
            from pdfminer.high_level import extract_text
        except ImportError:
            raise ImportError('`pdfminer` package not found, please install it with `pip install pdfminer.six`')
        super().__init__(file_path, headers=headers)
        self.parser = PDFMinerParser(extract_images=extract_images, concatenate_pages=concatenate_pages)

    def lazy_load(self) -> Iterator[Document]:
        """Lazily load documents."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, 'rb').read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)