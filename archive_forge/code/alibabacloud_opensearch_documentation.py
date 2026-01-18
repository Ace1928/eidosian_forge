import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Create alibaba cloud opensearch vector store instance.

        Args:
            documents: Documents to be inserted into the vector storage,
             should not be empty.
            embedding: Embedding function, Embedding function.
            config: Alibaba OpenSearch instance configuration.
            ids: Specify the ID for the inserted document. If left empty, the ID will be
             automatically generated based on the text content.
        Returns:
            AlibabaCloudOpenSearch: Alibaba cloud opensearch vector store instance.
        