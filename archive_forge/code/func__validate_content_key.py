import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _validate_content_key(self, data: Any) -> None:
    """Check if a content key is valid"""
    sample = data.first()
    if not isinstance(sample, dict):
        raise ValueError(f'Expected the jq schema to result in a list of objects (dict),                     so sample must be a dict but got `{type(sample)}`')
    if not self._is_content_key_jq_parsable and sample.get(self._content_key) is None:
        raise ValueError(f'Expected the jq schema to result in a list of objects (dict)                     with the key `{self._content_key}`')
    if self._is_content_key_jq_parsable and self.jq.compile(self._content_key).input(sample).text() is None:
        raise ValueError(f'Expected the jq schema to result in a list of objects (dict)                     with the key `{self._content_key}` which should be parsable by jq')