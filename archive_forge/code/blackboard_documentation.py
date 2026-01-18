import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
Parse the filename from an url.

        Args:
            url: Url to parse the filename from.

        Returns:
            The filename.

        Raises:
            ValueError: If the filename could not be parsed.
        