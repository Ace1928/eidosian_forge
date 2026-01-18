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
def _get_document_for_video_id(self, video_id: str, **kwargs: Any) -> Document:
    captions = self._get_transcripe_for_video_id(video_id)
    video_response = self.youtube_client.videos().list(part='id,snippet', id=video_id).execute()
    return Document(page_content=captions, metadata=video_response.get('items')[0])