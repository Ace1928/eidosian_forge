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
class PyMuPDFLoader(BasePDFLoader):
    """Load `PDF` files using `PyMuPDF`."""

    def __init__(self, file_path: str, *, headers: Optional[Dict]=None, extract_images: bool=False, **kwargs: Any) -> None:
        """Initialize with a file path."""
        try:
            import fitz
        except ImportError:
            raise ImportError('`PyMuPDF` package not found, please install it with `pip install pymupdf`')
        super().__init__(file_path, headers=headers)
        self.extract_images = extract_images
        self.text_kwargs = kwargs

    def _lazy_load(self, **kwargs: Any) -> Iterator[Document]:
        if kwargs:
            logger.warning(f'Received runtime arguments {kwargs}. Passing runtime args to `load` is deprecated. Please pass arguments during initialization instead.')
        text_kwargs = {**self.text_kwargs, **kwargs}
        parser = PyMuPDFParser(text_kwargs=text_kwargs, extract_images=self.extract_images)
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, 'rb').read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from parser.lazy_parse(blob)

    def load(self, **kwargs: Any) -> List[Document]:
        return list(self._lazy_load(**kwargs))

    def lazy_load(self) -> Iterator[Document]:
        yield from self._lazy_load()