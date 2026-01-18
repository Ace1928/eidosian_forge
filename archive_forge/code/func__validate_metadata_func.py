import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _validate_metadata_func(self, data: Any) -> None:
    """Check if the metadata_func output is valid"""
    sample = data.first()
    if self._metadata_func is not None:
        sample_metadata = self._metadata_func(sample, {})
        if not isinstance(sample_metadata, dict):
            raise ValueError(f'Expected the metadata_func to return a dict but got                         `{type(sample_metadata)}`')