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
class YoutubeLoader(BaseLoader):
    """Load `YouTube` transcripts."""

    def __init__(self, video_id: str, add_video_info: bool=False, language: Union[str, Sequence[str]]='en', translation: Optional[str]=None, transcript_format: TranscriptFormat=TranscriptFormat.TEXT, continue_on_failure: bool=False):
        """Initialize with YouTube video ID."""
        self.video_id = video_id
        self.add_video_info = add_video_info
        self.language = language
        if isinstance(language, str):
            self.language = [language]
        else:
            self.language = language
        self.translation = translation
        self.transcript_format = transcript_format
        self.continue_on_failure = continue_on_failure

    @staticmethod
    def extract_video_id(youtube_url: str) -> str:
        """Extract video id from common YT urls."""
        video_id = _parse_video_id(youtube_url)
        if not video_id:
            raise ValueError(f'Could not determine the video ID for the URL {youtube_url}')
        return video_id

    @classmethod
    def from_youtube_url(cls, youtube_url: str, **kwargs: Any) -> YoutubeLoader:
        """Given youtube URL, load video."""
        video_id = cls.extract_video_id(youtube_url)
        return cls(video_id, **kwargs)

    def load(self) -> List[Document]:
        """Load documents."""
        try:
            from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
        except ImportError:
            raise ImportError('Could not import youtube_transcript_api python package. Please install it with `pip install youtube-transcript-api`.')
        metadata = {'source': self.video_id}
        if self.add_video_info:
            video_info = self._get_video_info()
            metadata.update(video_info)
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(self.video_id)
        except TranscriptsDisabled:
            return []
        try:
            transcript = transcript_list.find_transcript(self.language)
        except NoTranscriptFound:
            transcript = transcript_list.find_transcript(['en'])
        if self.translation is not None:
            transcript = transcript.translate(self.translation)
        transcript_pieces = transcript.fetch()
        if self.transcript_format == TranscriptFormat.TEXT:
            transcript = ' '.join([t['text'].strip(' ') for t in transcript_pieces])
            return [Document(page_content=transcript, metadata=metadata)]
        elif self.transcript_format == TranscriptFormat.LINES:
            return [Document(page_content=t['text'].strip(' '), metadata=dict(((key, t[key]) for key in t if key != 'text'))) for t in transcript_pieces]
        else:
            raise ValueError('Unknown transcript format.')

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