import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Union
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _to_langchain_compatible_metadata(self, metadata: dict) -> dict:
    """Convert a dictionary to a compatible with langchain."""
    result = {}
    for key, value in metadata.items():
        if type(value) in {str, int, float}:
            result[key] = value
        else:
            result[key] = str(value)
    return result