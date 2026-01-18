import logging
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
import requests
from langchain_core.documents import Document
from tenacity import (
from langchain_community.document_loaders.base import BaseLoader
class ContentFormat(str, Enum):
    """Enumerator of the content formats of Confluence page."""
    EDITOR = 'body.editor'
    EXPORT_VIEW = 'body.export_view'
    ANONYMOUS_EXPORT_VIEW = 'body.anonymous_export_view'
    STORAGE = 'body.storage'
    VIEW = 'body.view'

    def get_content(self, page: dict) -> str:
        return page['body'][self.name.lower()]['value']