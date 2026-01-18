import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _get_message_metadata(self, message: dict, channel_name: str) -> dict:
    """Create and return metadata for a given message and channel."""
    timestamp = message.get('ts', '')
    user = message.get('user', '')
    source = self._get_message_source(channel_name, user, timestamp)
    return {'source': source, 'channel': channel_name, 'timestamp': timestamp, 'user': user}