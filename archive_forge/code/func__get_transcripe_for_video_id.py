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
def _get_transcripe_for_video_id(self, video_id: str) -> str:
    from youtube_transcript_api import NoTranscriptFound, YouTubeTranscriptApi
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        transcript = transcript_list.find_transcript([self.captions_language])
    except NoTranscriptFound:
        for available_transcript in transcript_list:
            transcript = available_transcript.translate(self.captions_language)
            continue
    transcript_pieces = transcript.fetch()
    return ' '.join([t['text'].strip(' ') for t in transcript_pieces])