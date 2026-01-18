import functools
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterator, Union
import yaml
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _remove_front_matter(self, content: str) -> str:
    """Remove front matter metadata from the given content."""
    if not self.collect_metadata:
        return content
    return self.FRONT_MATTER_REGEX.sub('', content)