import csv
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.helpers import detect_file_encodings
from langchain_community.document_loaders.unstructured import (


        Args:
            file_path: The path to the CSV file.
            mode: The mode to use when loading the CSV file.
              Optional. Defaults to "single".
            **unstructured_kwargs: Keyword arguments to pass to unstructured.
        