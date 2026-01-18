from typing import Callable, Union
from langchain_core.documents import Document
from langchain_community.docstore.base import Docstore
Search for a document.

        Args:
            search: search string

        Returns:
            Document if found, else error message.
        