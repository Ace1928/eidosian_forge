from __future__ import annotations
import os
from typing import (
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _dependable_mastodon_import() -> mastodon:
    try:
        import mastodon
    except ImportError:
        raise ImportError('Mastodon.py package not found, please install it with `pip install Mastodon.py`')
    return mastodon