from typing import Any, List
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors.base import (
Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        