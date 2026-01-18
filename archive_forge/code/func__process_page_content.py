from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def _process_page_content(self, page: pdfplumber.page.Page) -> str:
    """Process the page content based on dedupe."""
    if self.dedupe:
        return page.dedupe_chars().extract_text(**self.text_kwargs)
    return page.extract_text(**self.text_kwargs)