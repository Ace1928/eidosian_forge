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
def _get_video_info(self) -> dict:
    """Get important video information.

        Components are:
            - title
            - description
            - thumbnail url,
            - publish_date
            - channel_author
            - and more.
        """
    try:
        from pytube import YouTube
    except ImportError:
        raise ImportError('Could not import pytube python package. Please install it with `pip install pytube`.')
    yt = YouTube(f'https://www.youtube.com/watch?v={self.video_id}')
    video_info = {'title': yt.title or 'Unknown', 'description': yt.description or 'Unknown', 'view_count': yt.views or 0, 'thumbnail_url': yt.thumbnail_url or 'Unknown', 'publish_date': yt.publish_date.strftime('%Y-%m-%d %H:%M:%S') if yt.publish_date else 'Unknown', 'length': yt.length or 0, 'author': yt.author or 'Unknown'}
    return video_info