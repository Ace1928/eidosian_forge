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
class PDFPlumberLoader(BasePDFLoader):
    """Load `PDF` files using `pdfplumber`."""

    def __init__(self, file_path: str, text_kwargs: Optional[Mapping[str, Any]]=None, dedupe: bool=False, headers: Optional[Dict]=None, extract_images: bool=False) -> None:
        """Initialize with a file path."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError('pdfplumber package not found, please install it with `pip install pdfplumber`')
        super().__init__(file_path, headers=headers)
        self.text_kwargs = text_kwargs or {}
        self.dedupe = dedupe
        self.extract_images = extract_images

    def load(self) -> List[Document]:
        """Load file."""
        parser = PDFPlumberParser(text_kwargs=self.text_kwargs, dedupe=self.dedupe, extract_images=self.extract_images)
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, 'rb').read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        return parser.parse(blob)