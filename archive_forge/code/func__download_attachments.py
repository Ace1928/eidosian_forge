import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _download_attachments(self, attachments: List[str]) -> None:
    """Download all attachments.

        Args:
            attachments: List of attachments.
        """
    Path(self.folder_path).mkdir(parents=True, exist_ok=True)
    for attachment in attachments:
        self.download(attachment)