import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _get_documents(self, soup: Any) -> List[Document]:
    """Fetch content from page and return Documents.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            List of documents.
        """
    attachments = self._get_attachments(soup)
    self._download_attachments(attachments)
    documents = self._load_documents()
    return documents