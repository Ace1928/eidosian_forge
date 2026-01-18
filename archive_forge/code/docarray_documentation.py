from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores.utils import maximal_marginal_relevance

        Convert a DocArray document (which also might be a dict)
        to a langchain document format.

        DocArray document can contain arbitrary fields, so the mapping is done
        in the following way:

        page_content <-> content_field
        metadata <-> all other fields excluding
            tensors and embeddings (so float, int, string)

        Args:
            doc: DocArray document

        Returns:
            Document in langchain format

        Raises:
            ValueError: If the document doesn't contain the content field
        