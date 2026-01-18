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
class DocumentIntelligenceLoader(BasePDFLoader):
    """Loads a PDF with Azure Document Intelligence"""

    def __init__(self, file_path: str, client: Any, model: str='prebuilt-document', headers: Optional[Dict]=None) -> None:
        """
        Initialize the object for file processing with Azure Document Intelligence
        (formerly Form Recognizer).

        This constructor initializes a DocumentIntelligenceParser object to be used
        for parsing files using the Azure Document Intelligence API. The load method
        generates a Document node including metadata (source blob and page number)
        for each page.

        Parameters:
        -----------
        file_path : str
            The path to the file that needs to be parsed.
        client: Any
            A DocumentAnalysisClient to perform the analysis of the blob
        model : str
            The model name or ID to be used for form recognition in Azure.

        Examples:
        ---------
        >>> obj = DocumentIntelligenceLoader(
        ...     file_path="path/to/file",
        ...     client=client,
        ...     model="prebuilt-document"
        ... )
        """
        self.parser = DocumentIntelligenceParser(client=client, model=model)
        super().__init__(file_path, headers=headers)

    def load(self) -> List[Document]:
        """Load given path as pages."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load given path as pages."""
        blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)