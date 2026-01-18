from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.pydantic_v1.dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
def _get_channel_id(self, channel_name: str) -> str:
    request = self.youtube_client.search().list(part='id', q=channel_name, type='channel', maxResults=1)
    response = request.execute()
    channel_id = response['items'][0]['id']['channelId']
    return channel_id