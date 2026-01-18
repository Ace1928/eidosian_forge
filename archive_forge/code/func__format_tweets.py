from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _format_tweets(self, tweets: List[Dict[str, Any]], user_info: dict) -> Iterable[Document]:
    """Format tweets into a string."""
    for tweet in tweets:
        metadata = {'created_at': tweet['created_at'], 'user_info': user_info}
        yield Document(page_content=tweet['text'], metadata=metadata)