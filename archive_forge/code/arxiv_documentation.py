import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator

        Run Arxiv search and get the article texts plus the article meta information.
        See https://lukasschwab.me/arxiv.py/index.html#Search

        Returns: documents with the document.page_content in text format

        Performs an arxiv search, downloads the top k results as PDFs, loads
        them as Documents, and returns them.

        Args:
            query: a plaintext search query
        