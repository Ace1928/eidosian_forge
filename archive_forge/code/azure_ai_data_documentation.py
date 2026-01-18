from typing import Iterator, Optional
from langchain_community.docstore.document import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileIOLoader
A lazy loader for Documents.