from __future__ import annotations
import concurrent.futures
from pathlib import Path
from typing import Iterator, Literal, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import (
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.registry import get_parser
Create a concurrent generic document loader using a filesystem blob loader.

        Args:
            path: The path to the directory to load documents from.
            glob: The glob pattern to use to find documents.
            suffixes: The suffixes to use to filter documents. If None, all files
                      matching the glob will be loaded.
            exclude: A list of patterns to exclude from the loader.
            show_progress: Whether to show a progress bar or not (requires tqdm).
                           Proxies to the file system loader.
            parser: A blob parser which knows how to parse blobs into documents
            num_workers: Max number of concurrent workers to use.
            parser_kwargs: Keyword arguments to pass to the parser.
        