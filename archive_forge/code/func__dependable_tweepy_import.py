from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _dependable_tweepy_import() -> tweepy:
    try:
        import tweepy
    except ImportError:
        raise ImportError('tweepy package not found, please install it with `pip install tweepy`')
    return tweepy