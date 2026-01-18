from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import UUID, uuid4
import urllib.parse
from langsmith import schemas as ls_schemas
from langsmith import utils
from langsmith.client import ID_TYPE, RUN_TYPE_T, Client, _dumps_json
class _Baggage:
    """Baggage header information."""

    def __init__(self, metadata: Optional[Dict[str, str]]=None, tags: Optional[List[str]]=None):
        """Initialize the Baggage object."""
        self.metadata = metadata or {}
        self.tags = tags or []

    @classmethod
    def from_header(cls, header_value: Optional[str]) -> _Baggage:
        """Create a Baggage object from the given header value."""
        if not header_value:
            return cls()
        metadata = {}
        tags = []
        try:
            for item in header_value.split(','):
                key, value = item.split('=', 1)
                if key == f'{LANGSMITH_PREFIX}metadata':
                    metadata = json.loads(urllib.parse.unquote(value))
                elif key == f'{LANGSMITH_PREFIX}tags':
                    tags = urllib.parse.unquote(value).split(',')
        except Exception as e:
            logger.warning(f'Error parsing baggage header: {e}')
        return cls(metadata=metadata, tags=tags)

    def to_header(self) -> str:
        """Return the Baggage object as a header value."""
        items = []
        if self.metadata:
            serialized_metadata = _dumps_json(self.metadata)
            items.append(f'{LANGSMITH_PREFIX}metadata={urllib.parse.quote(serialized_metadata)}')
        if self.tags:
            serialized_tags = ','.join(self.tags)
            items.append(f'{LANGSMITH_PREFIX}tags={urllib.parse.quote(serialized_tags)}')
        return ','.join(items)